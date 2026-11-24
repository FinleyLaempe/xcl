//! GPU-owned textures in title memory, aliased as on the console.
//!
//! A console title places render targets, depth buffers and UAV textures at addresses of its
//! own memory and reuses that memory: transient targets of a frame share it, and it creates
//! several resource objects over the same address expecting one memory (seen on XDefiant: 16
//! placements at an occupied address in its first seconds, also of other formats and sizes).
//! A committed PC resource per placement breaks that – what the GPU writes through one never
//! shows in the other. So these textures are placed in *arenas*: D3D12 default heaps (video
//! memory) standing for ranges of the title's address space (32 MiB+), at the offset of their
//! address in it. Resources at one address then share memory as on the console (the same
//! description: the same PC layout, so the same texels). The CPU never sees these textures
//! (on the console it could, but GPU-owned targets are not CPU-written).
//!
//! Arenas cost video memory for whole ranges, and a placement straddling two made a third over
//! both while the two lived on with their resources: a retail game's streaming targets, packed
//! back to back over 3.5 GB of its addresses, made 15 GB of arenas in two seconds – past the
//! card's memory and the system's commit, and the device was removed. So the first choice is
//! the console's own model (`Pages`): a 64 KiB page of video memory for each 64 KiB page of
//! the title's address a target covers, and each target a reserved (tiled) resource whose tiles
//! are mapped onto the pages of its addresses. Targets at one address share pages, a page
//! exists once however many targets cover it, and what no target covers takes nothing.
//! Arenas remain for what cannot be tiled (an address not 64 KiB aligned, a resource the
//! driver does not make reserved: D32S8, MSAA): heaps of the placement's size, shared by
//! placements inside them, let go 2 s after their last resource (256 MiB ones: 2.8 GB of D32S8).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::util::HRESULT;

/// A heap of video memory for placements at title addresses `[base, base + size)`.
struct Arena<H> {
    base: usize,
    size: usize,
    /// the heap's alignment (a placement that needs more does not go in)
    align: usize,
    heap: H,
    /// resources placed in it
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
    id: u64,
}

/// The arenas, apart from D3D12 (`H`: the heap; numbers in the tests).
struct Arenas<H> {
    list: Vec<Arena<H>>,
    next_id: u64,
}

impl<H: Clone> Arenas<H> {
    const fn new() -> Self {
        Arenas { list: Vec::new(), next_id: 0 }
    }

    /// Arenas no resource used for `IDLE` go (their heap with the last reference).
    fn sweep(&mut self, now: std::time::Instant) {
        self.list.retain(|a| a.users > 0 || a.idle_since.is_none_or(|t| now.duration_since(t) < IDLE));
    }

    /// Where a placement of `bytes` at `va` (alignment `align`) goes: the newest arena holding
    /// it at an offset `align` allows, else a new arena of its size, `make(size, align)`.
    /// (heap, offset, arena id); None: no heap.
    fn take(&mut self, va: usize, bytes: usize, align: usize, now: std::time::Instant, make: impl FnOnce(usize, usize) -> Option<H>) -> Option<(H, usize, u64)> {
        self.sweep(now);
        let fits = |a: &Arena<H>| a.align >= align && a.base <= va && va + bytes <= a.base + a.size && (va - a.base) % align == 0;
        let i = match self.list.iter().rposition(fits) {
            Some(i) => i,
            None => {
                let size = bytes.div_ceil(align) * align;
                let heap = make(size, align)?;
                self.next_id += 1;
                self.list.push(Arena { base: va, size, align, heap, users: 0, idle_since: None, id: self.next_id });
                self.list.len() - 1
            }
        };
        let a = &mut self.list[i];
        a.users += 1;
        a.idle_since = None;
        Some((a.heap.clone(), va - a.base, a.id))
    }

    /// A resource in arena `id` released.
    fn release(&mut self, id: u64, now: std::time::Instant) {
        if let Some(a) = self.list.iter_mut().find(|a| a.id == id) {
            a.users = a.users.saturating_sub(1);
            if a.users == 0 {
                a.idle_since = Some(now);
            }
        }
    }

    fn bytes(&self) -> usize { self.list.iter().map(|a| a.size).sum() }
}

// D3D12 heaps are free-threaded
unsafe impl Send for Arenas<ID3D12Heap> {}

static ARENAS: Mutex<Arenas<ID3D12Heap>> = Mutex::new(Arenas::new());
/// resource → its arena's id
static ARENA_OF: Mutex<Option<std::collections::HashMap<usize, u64>>> = Mutex::new(None);
/// bytes of every arena made
static MADE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// (bytes of the arenas in use, bytes of all arenas made) – `XCL_ACTIVITY`
pub fn arena_bytes() -> (usize, usize) {
    (ARENAS.lock().unwrap().bytes(), MADE_BYTES.load(std::sync::atomic::Ordering::Relaxed))
}
static PLACED_TEXTURES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

/// A texture backed by one of the aliasable D3D12 heaps.
pub(super) fn is_placed_texture(resource: usize) -> bool {
    PLACED_TEXTURES.lock().unwrap().as_ref().is_some_and(|textures| textures.contains(&resource))
}

pub(super) fn forget(resource: usize) {
    let placed = PLACED_TEXTURES.lock().unwrap().as_mut().is_some_and(|textures| textures.remove(&resource));
    if placed {
        forget_pages(resource);
        forget_in_blocks(resource);
        if let Some(id) = ARENA_OF.lock().unwrap().as_mut().and_then(|m| m.remove(&resource)) {
            ARENAS.lock().unwrap().release(id, std::time::Instant::now());
        }
    }
}

/// Whether a console texture with these D3D12_RESOURCE_FLAGS is the GPU's (render target,
/// depth buffer, UAV).
pub fn gpu_owned(flags: u32) -> bool {
    flags & 0x7 != 0
}

/// A new arena's heap: `size` bytes of video memory for title addresses from `va`.
unsafe fn arena_heap(dev: &ID3D12Device, va: usize, size: usize, align: usize) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: size as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        // 64 KiB, or 4 MiB for MSAA targets
        Alignment: align as u64,
        Flags: D3D12_HEAP_FLAG_NONE,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = dev.CreateHeap(&desc, &mut heap) {
        crate::xlog_warn!("D3D12.X: video-memory arena for {va:#x}: CreateHeap: {e}");
        return None;
    }
    let made = MADE_BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_info!("D3D12.X: video-memory arena #{n} for title addresses {va:#x}..{:#x} ({made} bytes of arenas made so far)", va + size);
    }
    heap
}

/// bytes of a page – of the title's addresses and of a tile
const PAGE: usize = 0x1_0000;
/// pages of a video-memory pool heap (64 MiB)
const POOL_PAGES: u32 = 1024;
/// a page no target uses any more is kept this long for the next target at its address (with
/// what it holds – a title may make a target again over memory it rendered to)
const IDLE: std::time::Duration = std::time::Duration::from_secs(2);
/// a pool tile given back is not reused before this (the GPU may still be at work on what the
/// title released)
const COOL: std::time::Duration = std::time::Duration::from_secs(2);

/// A pool tile: (pool heap, tile in it).
type Tile = (u32, u32);

struct Page {
    tile: Tile,
    /// targets using the page
    users: u32,
    /// when the last of them went
    idle_since: Option<std::time::Instant>,
}

/// A page of `Pages::pages` (its title address) or of `Pages::extra_pages` (its key there).
#[derive(Clone, Copy)]
enum PageKey {
    Own(usize),
    Extra(usize, usize),
}

/// The pages of `idle` – those whose last user went, oldest first, each with when – that no
/// resource has used for `IDLE` go back to the pools: out of their map, their tile to `cooling`.
/// An entry whose page was used again since (in use now, or idle again from a later moment, with
/// a later entry) is dropped. Only the queue is looked at: every page of both maps, looked at at
/// every placement, grew with the square of what a title streams.
fn sweep_idle(
    idle: &mut std::collections::VecDeque<(PageKey, std::time::Instant)>,
    pages: &mut std::collections::HashMap<usize, Page>,
    extra_pages: &mut std::collections::HashMap<(usize, usize), Page>,
    cooling: &mut std::collections::VecDeque<(Tile, std::time::Instant)>,
    now: std::time::Instant,
) {
    while idle.front().is_some_and(|&(_, since)| now.duration_since(since) >= IDLE) {
        let Some((key, since)) = idle.pop_front() else { break };
        let still_idle = |p: &Page| p.idle_since == Some(since);
        let page = match key {
            PageKey::Own(a) if pages.get(&a).is_some_and(still_idle) => pages.remove(&a),
            PageKey::Extra(va, i) if extra_pages.get(&(va, i)).is_some_and(still_idle) => extra_pages.remove(&(va, i)),
            _ => None,
        };
        if let Some(p) = page {
            cooling.push_back((p.tile, now));
        }
    }
}

struct Target {
    va: usize,
    console_bytes: Option<u64>,
    total: u32,
    mapped: bool,
    used: Vec<usize>,
    used_extra: Vec<(usize, usize)>,
}

/// The title's pages of video memory for tiled textures (see the module's comment).
struct Pages {
    queue: ID3D12CommandQueue,
    /// signalled after each mapping; title queues wait for it before their next work
    fence: ID3D12Fence,
    value: u64,
    pools: Vec<ID3D12Heap>,
    free: Vec<Tile>,
    /// tiles given back, reusable after `COOL`
    cooling: std::collections::VecDeque<(Tile, std::time::Instant)>,
    /// title page address → its video memory
    pages: std::collections::HashMap<usize, Page>,
    /// extra / sub-64 KiB tiles keyed by `(va, tile_index)` so repeated placements at the same
    /// address share them without aliasing unrelated addresses
    extra_pages: std::collections::HashMap<(usize, usize), Page>,
    /// pages of both maps whose last user went, oldest first, with when (`sweep_idle`)
    idle: std::collections::VecDeque<(PageKey, std::time::Instant)>,
    /// resource → its mapping metadata and currently held pages / extra keys
    targets: std::collections::HashMap<usize, Target>,
    /// queue → the fence value it last waited for
    waited: std::collections::HashMap<usize, u64>,
}

// D3D12 objects are free-threaded
unsafe impl Send for Pages {}

/// device raw pointer → its pages (None inside: tiled tier < 2 on this device)
static PAGES: Mutex<std::collections::BTreeMap<usize, Option<Pages>>> = Mutex::new(std::collections::BTreeMap::new());

impl Pages {
    unsafe fn new(dev: &ID3D12Device) -> Option<Pages> {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).ok()?;
        // tier 2: unmapped tiles read as zero, writes to them go nowhere
        if o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
            crate::xlog_info!("D3D12.X: tiled resources tier {}: GPU-owned textures in arenas", o.TiledResourcesTier.0);
            return None;
        }
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).ok()?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        crate::xlog_info!("D3D12.X: textures in title memory: tiled resources on 64 KiB pages of video memory, one per page of the title's");
        Some(Pages {
            queue,
            fence,
            value: 0,
            pools: Vec::new(),
            free: Vec::new(),
            cooling: std::collections::VecDeque::new(),
            pages: std::collections::HashMap::new(),
            extra_pages: std::collections::HashMap::new(),
            idle: std::collections::VecDeque::new(),
            targets: std::collections::HashMap::new(),
            waited: std::collections::HashMap::new(),
        })
    }

    /// Pages no resource used for `IDLE` go back to the pools.
    fn sweep(&mut self, now: std::time::Instant) {
        sweep_idle(&mut self.idle, &mut self.pages, &mut self.extra_pages, &mut self.cooling, now);
    }

    unsafe fn take_tile(&mut self, dev: &ID3D12Device) -> Option<Tile> {
        if let Some(t) = self.free.pop() {
            return Some(t);
        }
        if self.cooling.front().is_some_and(|(_, at)| at.elapsed() >= COOL) {
            return self.cooling.pop_front().map(|(t, _)| t);
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: POOL_PAGES as u64 * PAGE as u64,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            // every kind of texture (resource heap tier 2: the only flags for both)
            Flags: D3D12_HEAP_FLAG_NONE,
        };
        let mut heap: Option<ID3D12Heap> = None;
        if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
            crate::xlog_warn!("D3D12.X: no video memory for more pages of title textures ({e})");
            return None;
        }
        self.pools.push(heap?);
        let h = self.pools.len() as u32 - 1;
        self.free.extend((1..POOL_PAGES).rev().map(|t| (h, t)));
        Some((h, 0))
    }

    /// Map every tile of `res` (`tiles[i]`: overall tile i) – one call per pool heap.
    unsafe fn map(&mut self, res: &ID3D12Resource, tiles: &[Tile]) {
        let mut heaps: Vec<u32> = tiles.iter().map(|t| t.0).collect();
        heaps.sort_unstable();
        heaps.dedup();
        for h in heaps {
            let ranges = heap_ranges(h, tiles);
            let flags: Vec<D3D12_TILE_RANGE_FLAGS> = ranges.iter().map(|r| if r.0 { D3D12_TILE_RANGE_FLAG_NONE } else { D3D12_TILE_RANGE_FLAG_SKIP }).collect();
            let offsets: Vec<u32> = ranges.iter().map(|r| r.1).collect();
            let counts: Vec<u32> = ranges.iter().map(|r| r.2).collect();
            let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
            let region = D3D12_TILE_REGION_SIZE { NumTiles: tiles.len() as u32, UseBox: false.into(), ..Default::default() };
            self.queue.UpdateTileMappings(
                res,
                1,
                Some(&coord),
                Some(&region),
                &self.pools[h as usize],
                ranges.len() as u32,
                Some(flags.as_ptr()),
                Some(offsets.as_ptr()),
                Some(counts.as_ptr()),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    /// Unmap every tile of `res` (`D3D12_TILE_RANGE_FLAG_NULL`) so its unshared tiles can return
    /// to the pool even while the title keeps the COM object alive.
    unsafe fn unmap(&mut self, res: &ID3D12Resource, total: u32) {
        let coord = D3D12_TILED_RESOURCE_COORDINATE::default();
        let region = D3D12_TILE_REGION_SIZE { NumTiles: total, UseBox: false.into(), ..Default::default() };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        self.queue.UpdateTileMappings(
            res,
            1,
            Some(&coord),
            Some(&region),
            None,
            1,
            Some(&flag),
            None,
            Some(&total),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        self.value += 1;
        let _ = self.queue.Signal(&self.fence, self.value);
    }

    fn release_tiles(&mut self, used: Vec<usize>, used_extra: Vec<(usize, usize)>, now: std::time::Instant, immediate_cool: bool) {
        for a in used {
            if let Some(p) = self.pages.get_mut(&a) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.pages.remove(&a) {
                            self.cooling.push_back((p.tile, now));
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Own(a), now));
                    }
                }
            }
        }
        for k in used_extra {
            if let Some(p) = self.extra_pages.get_mut(&k) {
                p.users = p.users.saturating_sub(1);
                if p.users == 0 {
                    if immediate_cool {
                        if let Some(p) = self.extra_pages.remove(&k) {
                            self.cooling.push_back((p.tile, now));
                        }
                    } else {
                        p.idle_since = Some(now);
                        self.idle.push_back((PageKey::Extra(k.0, k.1), now));
                    }
                }
            }
        }
    }

    /// Acquire/share all `total` tiles for `res` at `va` and map them on the `Pages` queue.
    unsafe fn bind(&mut self, dev: &ID3D12Device, res: &ID3D12Resource, va: usize, console_bytes: Option<u64>, total: u32) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        let (first, own) = (total > 0).then(|| owned_pages(va, console_bytes, total as usize)).flatten()?;
        let now = std::time::Instant::now();
        self.sweep(now);
        let mut tiles = Vec::with_capacity(total as usize);
        let mut used = Vec::with_capacity(own);
        let mut used_extra = Vec::new();
        for i in 0..total as usize {
            let tile = if i < own {
                let a = first + i * PAGE;
                let tile = match self.pages.get_mut(&a) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        Some(p.tile)
                    }
                    None => self.take_tile(dev).inspect(|&t| {
                        self.pages.insert(a, Page { tile: t, users: 1, idle_since: None });
                    }),
                };
                if tile.is_some() {
                    used.push(a);
                }
                tile
            } else {
                let k = (va, i);
                let tile = match self.extra_pages.get_mut(&k) {
                    Some(p) => {
                        p.users += 1;
                        p.idle_since = None;
                        Some(p.tile)
                    }
                    None => self.take_tile(dev).inspect(|&t| {
                        self.extra_pages.insert(k, Page { tile: t, users: 1, idle_since: None });
                    }),
                };
                if tile.is_some() {
                    used_extra.push(k);
                }
                tile
            };
            let Some(t) = tile else { break };
            tiles.push(t);
        }
        if tiles.len() < total as usize {
            self.release_tiles(used, used_extra, now, true);
            return None;
        }
        self.map(res, &tiles);
        Some((used, used_extra))
    }
}

/// The ranges of one UpdateTileMappings call for pool heap `heap` over a resource whose overall
/// tile i goes to `tiles[i]`: (mapped, tile offset in the heap, count) – runs of consecutive
/// tiles of the heap, and skipped runs for tiles of other heaps.
fn heap_ranges(heap: u32, tiles: &[Tile]) -> Vec<(bool, u32, u32)> {
    let mut out: Vec<(bool, u32, u32)> = Vec::new();
    for &(h, t) in tiles {
        let mine = h == heap;
        match out.last_mut() {
            Some(r) if mine && r.0 && r.1 + r.2 == t => r.2 += 1,
            Some(r) if !mine && !r.0 => r.2 += 1,
            _ => out.push((mine, if mine { t } else { 0 }, 1)),
        }
    }
    out
}

/// 64 KiB page boundaries `P` with `va <= P < va + bytes`, capped at `total` tiles:
/// `(first_page, count)`. Because every `P` lies strictly inside `[va, va + bytes)`, two
/// non-overlapping ranges never share a page here even when `va` is only 4 KiB aligned.
fn owned_pages(va: usize, console_bytes: Option<u64>, total: usize) -> Option<(usize, usize)> {
    match console_bytes {
        Some(b) => {
            let bytes = usize::try_from(b).ok()?;
            let first = va.checked_next_multiple_of(PAGE)?;
            let end = va.saturating_add(bytes);
            let n = if first < end { (end - first).div_ceil(PAGE) } else { 0 };
            Some((first, n.min(total)))
        }
        None => (va % PAGE == 0).then_some((va, total)),
    }
}

/// A reserved texture needs TILED in the format's support (`support2`) and, multisampled, a
/// tiled quality level at its sample count.
fn can_reserve(support2: i32, samples: u32, tiled_msaa_levels: u32) -> bool {
    support2 & D3D12_FORMAT_SUPPORT2_TILED.0 != 0 && (samples <= 1 || tiled_msaa_levels > 0)
}

/// Whether the device can make `desc` a reserved texture, asked before CreateReservedResource:
/// refused, it is an error in the debug layer (#940: R32G8X24_TYPELESS depth, 1 and 4 samples).
/// A query that fails says yes, and creation decides.
unsafe fn reservable(dev: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    let mut f = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: desc.Format, ..Default::default() };
    if dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut f as *mut _ as *mut c_void, std::mem::size_of_val(&f) as u32).is_err() {
        return true;
    }
    // a format without TILED is not reservable whatever the sample count's query says
    if !can_reserve(f.Support2.0, 1, 0) {
        return false;
    }
    let samples = desc.SampleDesc.Count;
    let mut m = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS { Format: desc.Format, SampleCount: samples, Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, NumQualityLevels: 0 };
    if samples > 1 && dev.CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &mut m as *mut _ as *mut c_void, std::mem::size_of_val(&m) as u32).is_err() {
        return true;
    }
    can_reserve(f.Support2.0, samples, m.NumQualityLevels)
}

/// A texture at `va` as a reserved resource on the title's shared 64 KiB pages
/// (`console_bytes`: its size on the console; tiles past the owned page boundaries are shared by
/// `(va, tile_index)` so repeated placements at the same address do not leak). None: it cannot be.
#[allow(clippy::too_many_arguments)]
unsafe fn place_on_pages(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    require_aligned: bool,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D {
        return None;
    }
    if (require_aligned || console_bytes.is_none()) && va % PAGE != 0 {
        return None;
    }
    // reserved textures have the PC's 64 KiB tiles
    let tiled = D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..*desc };
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let log = |why: &str| {
        if require_aligned && N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: GPU-owned texture at {va:#x} ({}x{}, format {}, {} samples) not reserved ({why}): in an arena", desc.Width, desc.Height, desc.Format.0, desc.SampleDesc.Count);
        }
    };
    if !reservable(dev, &tiled) {
        log("no tiled support for its format or sample count");
        return None;
    }
    let mut g = PAGES.lock().unwrap();
    let pages = g.entry(dev.as_raw() as usize).or_insert_with(|| Pages::new(dev)).as_mut()?;
    let hr = super::real::dev_create_reserved_resource(this, &tiled as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        log(&format!("{hr:#x}"));
        return None;
    }
    let raw = *ppv;
    let res = ID3D12Resource::from_raw_borrowed(&raw)?;
    let mut total = 0u32;
    dev.GetResourceTiling(res, Some(&mut total), None, None, None, 0, std::ptr::null_mut());
    let Some((used, used_extra)) = pages.bind(dev, res, va, console_bytes, total) else {
        drop(g);
        drop(windows::core::IUnknown::from_raw(raw));
        *ppv = std::ptr::null_mut();
        return None;
    };
    pages.targets.insert(raw as usize, Target { va, console_bytes, total, mapped: true, used, used_extra });
    drop(g);
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(raw as usize);
    Some(hr)
}

/// Unmap a placed texture superseded by a placement with a different address range, returning its
/// unshared tiles (`extra_pages` and unshared tail pages) to the cooling pool immediately.
pub(super) unsafe fn unmap_superseded(res: &ID3D12Resource) {
    let key = res.as_raw() as usize;
    if release_in_blocks(key) {
        return;
    }
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.get_mut(&key) else { continue };
        if !t.mapped {
            return;
        }
        let total = t.total;
        let used = std::mem::take(&mut t.used);
        let used_extra = std::mem::take(&mut t.used_extra);
        t.mapped = false;
        pages.unmap(res, total);
        pages.release_tiles(used, used_extra, now, true);
        return;
    }
}

/// What `remap_if_unmapped` did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Remap {
    /// the texture has its memory (again)
    Mapped,
    /// not now (no device, not ours, no tiles to bind): it can be mapped later
    Failed,
    /// its block-heap run went to another texture: it stays placed there, aliasing that one
    RunTaken,
}

/// Re-map a previously unmapped placed texture when the newer placement shadowing it is released.
pub(super) unsafe fn remap_if_unmapped(res: &ID3D12Resource) -> Remap {
    if let Some(back) = retake_in_blocks(res.as_raw() as usize) {
        if !back {
            return Remap::RunTaken;
        }
        // a run found cooled or free may have held another texture meanwhile (taken and given
        // back again): the title's unchanged memory is uploaded into it again
        super::title_textures::forget_content(res.as_raw() as usize);
        return Remap::Mapped;
    }
    let mut dev: Option<ID3D12Device> = None;
    if res.GetDevice(&mut dev).is_err() {
        return Remap::Failed;
    }
    let Some(dev) = dev else { return Remap::Failed };
    let key = res.as_raw() as usize;
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return Remap::Failed };
    let Some((va, console_bytes, total, mapped)) = pages.targets.get(&key).map(|t| (t.va, t.console_bytes, t.total, t.mapped)) else {
        return Remap::Failed;
    };
    if mapped {
        return Remap::Mapped;
    }
    let Some((used, used_extra)) = pages.bind(&dev, res, va, console_bytes, total) else {
        return Remap::Failed;
    };
    if let Some(t) = pages.targets.get_mut(&key) {
        t.used = used;
        t.used_extra = used_extra;
        t.mapped = true;
    }
    Remap::Mapped
}

/// A resource released: its pages are no longer its.
fn forget_pages(resource: usize) {
    let mut g = PAGES.lock().unwrap();
    let now = std::time::Instant::now();
    for pages in g.values_mut().flatten() {
        let Some(t) = pages.targets.remove(&resource) else { continue };
        if t.mapped {
            pages.release_tiles(t.used, t.used_extra, now, false);
        }
        return;
    }
}

/// Before GPU work on `queue`: it waits for the pages mapped since its last work on that device.
pub(super) unsafe fn before_execute(queue: &ID3D12CommandQueue) {
    let mut dev: Option<ID3D12Device> = None;
    if queue.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let mut g = PAGES.lock().unwrap();
    let Some(Some(pages)) = g.get_mut(&(dev.as_raw() as usize)) else { return };
    let key = queue.as_raw() as usize;
    if pages.waited.get(&key).copied().unwrap_or(0) < pages.value {
        let _ = queue.Wait(&pages.fence, pages.value);
        pages.waited.insert(key, pages.value);
    }
}

#[cfg(test)]
pub(super) static VRAM_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(super) fn page_bytes_for(device: usize) -> (usize, usize) {
    let g = PAGES.lock().unwrap();
    let Some(Some(p)) = g.get(&device) else { return (0, 0) };
    ((p.pages.len() + p.extra_pages.len()) * PAGE, p.pools.len() * POOL_PAGES as usize * PAGE)
}

/// (bytes of pages in use by tiled textures and of block-heap runs, bytes of pool and block
/// heaps) – `XCL_ACTIVITY`
pub fn page_bytes() -> (usize, usize) {
    let blocks = block_bytes();
    let g = PAGES.lock().unwrap();
    g.values().flatten().fold(blocks, |(used, pools), p| {
        (used + (p.pages.len() + p.extra_pages.len()) * PAGE, pools + p.pools.len() * POOL_PAGES as usize * PAGE)
    })
}

/// Heaps of every kind of resource in one (resource heap tier 2) – else no arenas.
fn tier2(dev: &ID3D12Device) -> bool {
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.ResourceHeapTier.0 >= 2 }
}

/// A sampled texture of the title at `va` (`desc`: the PC description, `console_bytes` its size on
/// the console) backed by shared 64 KiB video-memory pages; None when reserved placement is not
/// available (the caller falls back to a committed resource).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place_pages(
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    console_bytes: Option<u64>,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) || std::env::var_os("XCL_VRAM_ARENAS").is_some() {
        return None;
    }
    if let Some(hr) = place_in_blocks(dev, this, va, desc, state, clear, riid, ppv) {
        return Some(hr);
    }
    place_on_pages(dev, this, va, desc, state, clear, console_bytes, false, riid, ppv)
}

/// tiles (64 KiB) of a block heap (64 MiB)
const BLOCK_TILES: usize = 1024;
/// a run given back is not reused before this: as long as a page waits (`IDLE`, then `COOL`)
/// before another texture gets it – the GPU may still be at work on what the title released
const BLOCK_COOL: std::time::Duration = std::time::Duration::from_secs(4);
/// after a block heap could not be made, no new one is tried for this long (the textures take
/// pages meanwhile)
const HEAP_BACKOFF: std::time::Duration = std::time::Duration::from_secs(5);

/// A run of tiles of the block heaps, shared by the textures placed at one address with one size.
struct Run {
    heap: usize,
    first: usize,
    users: u32,
}

/// A texture in the block heaps: its address and tiles, its run, whether it holds the run.
#[derive(Clone, Copy)]
struct InBlocks {
    va: usize,
    tiles: usize,
    heap: usize,
    first: usize,
    held: bool,
}

/// The block heaps of a device and the runs of tiles in them, apart from D3D12 (`H`: the heap;
/// numbers in the tests).
struct Blocks<H> {
    heaps: Vec<H>,
    /// tiles handed out of the last heap
    tail: usize,
    /// cooled runs by length: (heap, first tile)
    free: std::collections::BTreeMap<usize, Vec<(usize, usize)>>,
    /// runs given back, reusable after `BLOCK_COOL`: (heap, first tile, tiles, when)
    cooling: std::collections::VecDeque<(usize, usize, usize, std::time::Instant)>,
    /// (address, tiles) → its run
    runs: std::collections::HashMap<(usize, usize), Run>,
    textures: std::collections::HashMap<usize, InBlocks>,
    /// tiles in runs
    used: usize,
    /// no new heap is tried before this (one could not be made)
    no_heap_until: Option<std::time::Instant>,
}

impl<H> Blocks<H> {
    fn new() -> Self {
        Blocks { heaps: Vec::new(), tail: 0, free: Default::default(), cooling: Default::default(), runs: Default::default(), textures: Default::default(), used: 0, no_heap_until: None }
    }

    /// `n` free tiles: a cooled run (the rest of a longer one stays free), else the next tiles of
    /// the last heap, else a new heap from `make` (None: no heap; then none is tried again for
    /// `HEAP_BACKOFF`).
    fn alloc(&mut self, n: usize, now: std::time::Instant, make: impl FnOnce() -> Option<H>) -> Option<(usize, usize)> {
        while self.cooling.front().is_some_and(|c| now.duration_since(c.3) >= BLOCK_COOL) {
            let (heap, first, len, _) = self.cooling.pop_front()?;
            self.free.entry(len).or_default().push((heap, first));
        }
        if let Some(len) = self.free.range(n..).find(|(_, v)| !v.is_empty()).map(|(&len, _)| len) {
            let runs = self.free.get_mut(&len)?;
            let (heap, first) = runs.pop()?;
            if runs.is_empty() {
                self.free.remove(&len);
            }
            if len > n {
                self.free.entry(len - n).or_default().push((heap, first + n));
            }
            return Some((heap, first));
        }
        if self.heaps.is_empty() || self.tail + n > BLOCK_TILES {
            if self.no_heap_until.is_some_and(|t| now < t) {
                return None;
            }
            let Some(heap) = make() else {
                self.no_heap_until = Some(now + HEAP_BACKOFF);
                return None;
            };
            if self.tail < BLOCK_TILES && !self.heaps.is_empty() {
                self.free.entry(BLOCK_TILES - self.tail).or_default().push((self.heaps.len() - 1, self.tail));
            }
            self.heaps.push(heap);
            self.tail = 0;
        }
        self.tail += n;
        Some((self.heaps.len() - 1, self.tail - n))
    }

    /// The run of one more texture of `n` tiles at `va`: (heap, first tile).
    fn take(&mut self, va: usize, n: usize, now: std::time::Instant, make: impl FnOnce() -> Option<H>) -> Option<(usize, usize)> {
        if let Some(r) = self.runs.get_mut(&(va, n)) {
            r.users += 1;
            return Some((r.heap, r.first));
        }
        let (heap, first) = self.alloc(n, now, make)?;
        self.runs.insert((va, n), Run { heap, first, users: 1 });
        self.used += n;
        Some((heap, first))
    }

    /// One texture of `n` tiles at `va` fewer: the run cools once none is left.
    fn give_back(&mut self, va: usize, n: usize, now: std::time::Instant) {
        let Some(r) = self.runs.get_mut(&(va, n)) else { return };
        r.users -= 1;
        if r.users == 0 {
            let r = self.runs.remove(&(va, n)).unwrap();
            self.cooling.push_back((r.heap, r.first, n, now));
            self.used -= n;
        }
    }

    /// Texture `res` superseded or released: it gives its run back.
    fn release(&mut self, res: usize, now: std::time::Instant) {
        if let Some(t) = self.textures.get_mut(&res).filter(|t| t.held) {
            t.held = false;
            let t = *t;
            self.give_back(t.va, t.tiles, now);
        }
    }

    /// Texture `res` uncovered: whether it holds its run again – not when another texture took the
    /// run meanwhile (it is still placed there).
    fn retake(&mut self, res: usize) -> bool {
        let Some(t) = self.textures.get(&res).copied() else { return false };
        if t.held {
            return true;
        }
        let back = match self.runs.get_mut(&(t.va, t.tiles)) {
            Some(r) if (r.heap, r.first) == (t.heap, t.first) => {
                r.users += 1;
                true
            }
            Some(_) => false,
            None => {
                let cooling = self.cooling.iter().position(|c| (c.0, c.1, c.2) == (t.heap, t.first, t.tiles));
                let free = self.free.get(&t.tiles).and_then(|v| v.iter().position(|&f| f == (t.heap, t.first)));
                let found = match (cooling, free) {
                    (Some(i), _) => self.cooling.remove(i).is_some(),
                    (None, Some(i)) => {
                        let runs = self.free.get_mut(&t.tiles).unwrap();
                        runs.swap_remove(i);
                        if runs.is_empty() {
                            self.free.remove(&t.tiles);
                        }
                        true
                    }
                    _ => false,
                };
                if found {
                    self.runs.insert((t.va, t.tiles), Run { heap: t.heap, first: t.first, users: 1 });
                    self.used += t.tiles;
                }
                found
            }
        };
        if back {
            self.textures.insert(res, InBlocks { held: true, ..t });
        }
        back
    }
}

/// The tiles a texture the PC places in `bytes` aligned to `align` takes in the block heaps
/// (None: none – nothing, more than a heap, or aligned to more than a tile, as MSAA is).
fn block_tiles(bytes: u64, align: u64) -> Option<usize> {
    (bytes > 0 && bytes <= (BLOCK_TILES * PAGE) as u64 && align > 0 && PAGE as u64 % align == 0).then(|| (bytes as usize).div_ceil(PAGE))
}

// D3D12 heaps are free-threaded
unsafe impl Send for Blocks<ID3D12Heap> {}

#[cfg(test)]
thread_local! {
    /// tests of the pages: this thread's sampled textures take them, not the block heaps
    pub(super) static PAGES_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// device raw pointer → its block heaps
static BLOCKS: Mutex<std::collections::BTreeMap<usize, Blocks<ID3D12Heap>>> = Mutex::new(std::collections::BTreeMap::new());

/// A block heap (64 MiB of video memory for sampled textures).
unsafe fn block_heap(dev: &ID3D12Device) -> Option<ID3D12Heap> {
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: (BLOCK_TILES * PAGE) as u64,
        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
    };
    let mut heap: Option<ID3D12Heap> = None;
    if let Err(e) = super::title_textures::own_work(|| dev.CreateHeap(&desc, &mut heap)) {
        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            crate::xlog_warn!("D3D12.X: no video memory for a block heap of sampled textures ({e}): they take pages ({n} times so far; tried again after {} s)", HEAP_BACKOFF.as_secs());
        }
        return None;
    }
    heap
}

/// A sampled texture at `va` as a placed resource in a run of tiles of the block heaps, not a
/// reserved one on the pages. Under vkd3d-proton a reserved texture is a Vulkan sparse image,
/// and on NVIDIA's Linux driver making one costs a kernel call that grows with the sparse images
/// alive (0.1 ms at 10 000, 1.5–4 ms at 60 000; a placed texture 1.4 us at any count): a retail
/// game keeps ~60 000 streamed textures alive while it loads its world and spent 1–1.9 s of every
/// 2 s making them (#1268). Most of them sit at 4 KiB-aligned addresses, where the PC cannot
/// place them at their address's offset in a heap standing for the title's addresses, so the run
/// is anywhere: textures at one address with one size share it (as they shared pages), and a
/// texture superseded by a newer one over its memory (`title_textures`, `unmap_superseded`) gives
/// it back, so the runs follow the textures in use as the pages do. None: not placed here.
#[allow(clippy::too_many_arguments)]
unsafe fn place_in_blocks(
    dev: &ID3D12Device,
    this: *mut c_void,
    va: usize,
    desc: &D3D12_RESOURCE_DESC,
    state: u32,
    clear: *const c_void,
    riid: *const windows_sys::core::GUID,
    ppv: *mut *mut c_void,
) -> Option<HRESULT> {
    if !takes_blocks(desc) {
        return None;
    }
    #[cfg(test)]
    if PAGES_ONLY.with(|p| p.get()) {
        return None;
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    let tiles = block_tiles(info.SizeInBytes, info.Alignment)?;
    let now = std::time::Instant::now();
    let key = dev.as_raw() as usize;
    // the run is taken under the lock, the texture made outside it (other threads' textures are
    // made meanwhile); the run counts this texture from here on
    let (heap_obj, heap, first) = {
        let mut g = BLOCKS.lock().unwrap();
        let b = g.entry(key).or_insert_with(Blocks::new);
        let (heap, first) = b.take(va, tiles, now, || block_heap(dev))?;
        (b.heaps[heap].clone(), heap, first)
    };
    let hr = super::real::dev_create_placed_resource(this, heap_obj.as_raw() as usize, first * PAGE, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    let mut g = BLOCKS.lock().unwrap();
    let b = g.get_mut(&key)?;
    if hr < 0 || (*ppv).is_null() {
        b.give_back(va, tiles, now);
        return None;
    }
    b.textures.insert(*ppv as usize, InBlocks { va, tiles, heap, first, held: true });
    drop(g);
    // aliased by address like the pages' textures
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    Some(hr)
}

/// Whether a sampled texture goes in the block heaps: only in the layout the driver picks.
fn takes_blocks(desc: &D3D12_RESOURCE_DESC) -> bool {
    desc.Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN
}

/// A texture of the block heaps superseded (true) or not one of them.
fn release_in_blocks(res: usize) -> bool {
    let mut g = BLOCKS.lock().unwrap();
    let b = g.values_mut().find(|b| b.textures.contains_key(&res));
    b.map(|b| b.release(res, std::time::Instant::now())).is_some()
}

/// A texture of the block heaps uncovered: whether it holds its run again (None: not one of them).
fn retake_in_blocks(res: usize) -> Option<bool> {
    let mut g = BLOCKS.lock().unwrap();
    g.values_mut().find(|b| b.textures.contains_key(&res)).map(|b| b.retake(res))
}

/// A released texture gives its run back.
fn forget_in_blocks(res: usize) {
    let mut g = BLOCKS.lock().unwrap();
    if let Some(b) = g.values_mut().find(|b| b.textures.contains_key(&res)) {
        b.release(res, std::time::Instant::now());
        b.textures.remove(&res);
    }
}

/// The run of a texture of the block heaps: (heap, first tile).
#[cfg(test)]
pub(super) fn block_run_of(res: usize) -> Option<(usize, usize)> {
    BLOCKS.lock().unwrap().values().find_map(|b| b.textures.get(&res).filter(|t| t.held).map(|t| (t.heap, t.first)))
}

/// Tests: the runs given back so far are cooled (as if their time had passed).
#[cfg(test)]
pub(super) fn cool_blocks_for_test() {
    for b in BLOCKS.lock().unwrap().values_mut() {
        for c in b.cooling.iter_mut() {
            c.3 = c.3.checked_sub(BLOCK_COOL).unwrap_or(c.3);
        }
    }
}

/// (bytes of runs in use, bytes of block heaps)
fn block_bytes() -> (usize, usize) {
    let g = BLOCKS.lock().unwrap();
    g.values().fold((0, 0), |(used, all), b| (used + b.used * PAGE, all + b.heaps.len() * BLOCK_TILES * PAGE))
}

/// A GPU-owned texture of the title at `va` (`desc`: the PC description, `console_bytes` its
/// size on the console): on the title's pages there, else in the arena for that address (None:
/// neither – the caller makes a resource of its own).
#[allow(clippy::too_many_arguments)]
pub unsafe fn place(this: *mut c_void, va: usize, desc: &D3D12_RESOURCE_DESC, state: u32, clear: *const c_void, console_bytes: Option<u64>, riid: *const windows_sys::core::GUID, ppv: *mut *mut c_void) -> Option<HRESULT> {
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    if !tier2(dev) {
        return None;
    }
    if std::env::var_os("XCL_VRAM_ARENAS").is_none() {
        if let Some(hr) = place_on_pages(dev, this, va, desc, state, clear, console_bytes, true, riid, ppv) {
            return Some(hr);
        }
    }
    let info = dev.GetResourceAllocationInfo(0, &[*desc]);
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || va as u64 % info.Alignment.max(1) != 0 {
        return None;
    }
    let align = (info.Alignment as usize).max(PAGE);
    let now = std::time::Instant::now();
    let (heap, offset, id) = ARENAS.lock().unwrap().take(va, info.SizeInBytes as usize, align, now, |size, align| arena_heap(dev, va, size, align))?;
    let hr = super::real::dev_create_placed_resource(this, heap.as_raw() as usize, offset, desc as *const _ as usize, state as usize, clear as usize, riid, ppv);
    if hr < 0 || (*ppv).is_null() {
        ARENAS.lock().unwrap().release(id, now);
        return None;
    }
    PLACED_TEXTURES.lock().unwrap().get_or_insert_with(HashSet::new).insert(*ppv as usize);
    ARENA_OF.lock().unwrap().get_or_insert_with(std::collections::HashMap::new).insert(*ppv as usize, id);
    Some(hr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const MIB: usize = 1 << 20;

    /// Runs of the block heaps: one per (address, tiles), shared there; a run given back is
    /// reused after `BLOCK_COOL` (a longer one split), not after the pages' `COOL` alone, a heap's
    /// end left over is reused, and a new heap comes when nothing fits.
    #[test]
    fn block_runs_are_shared_by_address_and_reused_after_cooling() {
        let (t, a0) = (Instant::now(), 0x40_0000_1000usize);
        let mut b: Blocks<u32> = Blocks::new();
        let mut n = 0u32;
        let mut make = || {
            n += 1;
            Some(n - 1)
        };
        assert_eq!(b.take(a0, 3, t, &mut make), Some((0, 0)));
        assert_eq!(b.take(a0, 3, t, &mut make), Some((0, 0)), "the same address and size: the same run");
        assert_eq!(b.take(a0, 4, t, &mut make), Some((0, 3)), "another size: a run of its own");
        assert_eq!(b.take(a0 + 0x1000, 3, t, &mut make), Some((0, 7)), "another address: a run of its own");
        assert_eq!(b.used, 10);
        b.give_back(a0, 3, t);
        assert_eq!(b.used, 10, "still used by the other texture there");
        b.give_back(a0, 3, t);
        b.give_back(a0, 4, t);
        assert_eq!(b.take(a0 + 0x2000, 2, t + COOL, &mut make), Some((0, 10)), "not reused before BLOCK_COOL");
        // cooled: runs of 3 at 0 and 4 at 3
        assert_eq!(b.take(a0 + 0x3000, 2, t + BLOCK_COOL, &mut make), Some((0, 0)), "the shortest cooled run that fits, split");
        assert_eq!(b.take(a0 + 0x4000, 3, t + BLOCK_COOL, &mut make), Some((0, 3)));
        assert_eq!(b.take(a0 + 0x5000, 1, t + BLOCK_COOL, &mut make), Some((0, 6)), "the rest of a split run");
        assert_eq!(b.take(a0 + 0x6000, BLOCK_TILES - 12, t + BLOCK_COOL, &mut make), Some((0, 12)));
        assert_eq!(b.take(a0 + 0x7000, 2, t + BLOCK_COOL, &mut make), Some((1, 0)), "a new heap");
        assert_eq!(b.take(a0 + 0x8000, BLOCK_TILES, t + BLOCK_COOL, || None), None, "no heap and nothing free");
        assert_eq!(b.heaps.len(), 2);
    }

    /// A heap that cannot be made: no run (the texture takes pages), and no new heap is tried for
    /// `HEAP_BACKOFF` – runs that fit in what is left are still handed out meanwhile; a heap filled
    /// exactly leaves nothing over when the next one comes.
    #[test]
    fn no_new_block_heap_is_tried_for_a_while_after_one_could_not_be_made() {
        let (t, a0) = (Instant::now(), 0x41_0000_0000usize);
        let mut b: Blocks<u32> = Blocks::new();
        let mut tries = 0;
        assert_eq!(b.take(a0, 4, t, || { tries += 1; None }), None, "no first heap");
        assert!(b.runs.is_empty() && b.used == 0 && b.heaps.is_empty());
        assert_eq!(b.take(a0, 4, t + HEAP_BACKOFF / 2, || { tries += 1; Some(0) }), None, "not tried again yet");
        assert_eq!(tries, 1);
        assert_eq!(b.take(a0, BLOCK_TILES, t + HEAP_BACKOFF, || { tries += 1; Some(0) }), Some((0, 0)), "tried again: a whole heap");
        assert_eq!(tries, 2);
        let t2 = t + HEAP_BACKOFF;
        assert_eq!(b.take(a0 + 0x1000, 1, t2, || None), None, "the next heap cannot be made");
        b.give_back(a0, BLOCK_TILES, t2);
        assert_eq!(b.take(a0 + 0x2000, 4, t2 + BLOCK_COOL, || panic!("no heap is tried meanwhile")), Some((0, 0)), "a cooled run that fits is still handed out");
        assert_eq!(b.take(a0 + 0x3000, BLOCK_TILES, t2 + BLOCK_COOL, || panic!("no heap is tried meanwhile")), None, "the rest is too short, and no heap yet");
        assert_eq!(b.take(a0 + 0x3000, BLOCK_TILES, t2 + HEAP_BACKOFF, || Some(1)), Some((1, 0)), "a new heap after the back-off");
        assert_eq!(b.heaps, vec![0, 1]);
        let free: Vec<_> = b.free.iter().filter(|(_, v)| !v.is_empty()).map(|(&len, v)| (len, v.clone())).collect();
        assert_eq!(free, vec![(BLOCK_TILES - 4, vec![(0, 4)])], "only the split run's rest: nothing over from the full heap's end");
    }

    /// A superseded texture gives its run back and gets it again when uncovered – unless another
    /// texture took the run meanwhile; a second texture at the address keeps the run meanwhile.
    #[test]
    fn a_superseded_texture_gets_its_run_back_unless_another_took_it() {
        let (t, a0) = (Instant::now(), 0x40_0000_0000usize);
        let mut b: Blocks<u32> = Blocks::new();
        let place = |b: &mut Blocks<u32>, res: usize, va: usize, n: usize, now: Instant| {
            let (heap, first) = b.take(va, n, now, || Some(0)).unwrap();
            b.textures.insert(res, InBlocks { va, tiles: n, heap, first, held: true });
            (heap, first)
        };
        let run = place(&mut b, 1, a0, 4, t);
        b.release(1, t);
        assert!(b.retake(1) && b.runs[&(a0, 4)].users == 1, "uncovered within BLOCK_COOL: its run");
        place(&mut b, 2, a0, 4, t);
        b.release(1, t);
        assert_eq!(b.runs[&(a0, 4)].users, 1, "the other texture there keeps the run");
        assert!(b.retake(1) && b.runs[&(a0, 4)].users == 2);
        b.release(1, t);
        b.release(2, t);
        assert!(!b.runs.contains_key(&(a0, 4)) && b.used == 0);
        assert_eq!(place(&mut b, 3, a0 + PAGE, 4, t + BLOCK_COOL), run, "the cooled run goes to another texture");
        assert!(!b.retake(1), "taken: no run to get back");
        assert!(!b.textures[&1].held && b.retake(3));
    }

    /// A run given back and cooled into the free runs is got back by its texture when nobody took
    /// it; once a shorter texture took its front, the texture does not get it, and what is free
    /// stays as it was. Unknown textures and runs, and a second release, change nothing.
    #[test]
    fn retake_finds_a_free_run_and_leaves_a_split_one() {
        let (t, a0) = (Instant::now(), 0x42_0000_0000usize);
        let mut b: Blocks<u32> = Blocks::new();
        let place = |b: &mut Blocks<u32>, res: usize, va: usize, n: usize, now: Instant| {
            let (heap, first) = b.take(va, n, now, || Some(0)).unwrap();
            b.textures.insert(res, InBlocks { va, tiles: n, heap, first, held: true });
            (heap, first)
        };
        let run = place(&mut b, 1, a0, 4, t);
        place(&mut b, 2, a0 + PAGE, 2, t);
        b.release(1, t);
        b.release(1, t);
        b.give_back(a0 + 3 * PAGE, 4, t);
        b.release(99, t);
        assert_eq!(b.used, 2, "a second release and unknown ones change nothing");
        // cooled into the free runs (an allocation too long for it moves them there)
        place(&mut b, 3, a0 + 2 * PAGE, 5, t + BLOCK_COOL);
        assert!(b.cooling.is_empty() && b.free.get(&4).is_some_and(|v| v.contains(&run)), "a's run is free");
        assert!(!b.retake(99), "an unknown texture");
        assert!(b.retake(1), "nobody took it: a gets it back from the free runs");
        assert_eq!((b.runs[&(a0, 4)].heap, b.runs[&(a0, 4)].first), run);
        assert_eq!(b.used, 11);
        assert!(b.free.values().all(|v| !v.contains(&run)), "no longer free");
        assert!(b.free.values().all(|v| !v.is_empty()), "no length is kept without a run");
        // given back again, cooled, and a shorter texture takes its front
        b.release(1, t + BLOCK_COOL);
        assert_eq!(place(&mut b, 4, a0 + 4 * PAGE, 3, t + 2 * BLOCK_COOL), run, "the run split");
        let free_before = b.free.clone();
        assert!(!b.retake(1), "its front is another texture's");
        assert_eq!(b.free, free_before, "the free runs are as they were");
        assert!(!b.textures[&1].held && b.used == 10);
    }

    /// Any texture up to a heap whose PC alignment is at most a tile takes the block heaps.
    #[test]
    fn block_tiles_take_textures_up_to_a_heap_aligned_to_a_tile_or_less() {
        assert_eq!(block_tiles(0x1_6000, PAGE as u64), Some(2));
        assert_eq!(block_tiles(0x1000, 0x1000), Some(1));
        assert_eq!(block_tiles((BLOCK_TILES * PAGE) as u64, PAGE as u64), Some(BLOCK_TILES));
        assert_eq!(block_tiles((BLOCK_TILES * PAGE) as u64 + 1, PAGE as u64), None, "more than a heap");
        assert_eq!(block_tiles(4 * MIB as u64, 4 * MIB as u64), None, "MSAA: 4 MiB aligned");
        assert_eq!(block_tiles(0, PAGE as u64), None);
        assert_eq!(block_tiles(u64::MAX, PAGE as u64), None);
    }

    /// On the GPU: sampled textures at 4 KiB-aligned addresses and a cube take the block heaps, not
    /// pages; a second texture at an address with the same size shares its run; neighbours keep
    /// their own texels; a superseded texture gives its run back and gets it again.
    #[test]
    fn sampled_textures_take_block_heaps_not_pages() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            // 512x512 RGBA8, one level (1 MiB), and a 64x64 cube with mips
            let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R8G8B8A8_UNORM, SampleDesc: sample, ..Default::default() };
            let cube = D3D12_RESOURCE_DESC { Width: 64, Height: 64, DepthOrArraySize: 6, MipLevels: 7, ..desc };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place_pages(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32, std::ptr::null(), Some(MIB as u64), riid, &mut p);
                (hr.is_some_and(|hr| hr >= 0) && !p.is_null()).then(|| ID3D12Resource::from_raw(p)).expect("texture")
            };
            let key = dev.as_raw() as usize;
            let held = |r: &ID3D12Resource| BLOCKS.lock().unwrap().get(&key).and_then(|b| b.textures.get(&(r.as_raw() as usize)).map(|t| t.held));
            let pages0 = page_bytes_for(key).0;
            let base = 0x5a_0000_3000usize;
            let textures: Vec<ID3D12Resource> = (0..4).map(|k| place_at(base + k * MIB, &desc)).collect();
            let c = place_at(base + 8 * MIB, &cube);
            assert!(textures.iter().chain([&c]).all(|t| held(t) == Some(true) && is_placed_texture(t.as_raw() as usize)), "in the block heaps, aliased by address");
            assert_eq!(page_bytes_for(key).0, pages0, "no pages");
            let again = place_at(base, &desc);
            let run = |r: &ID3D12Resource| BLOCKS.lock().unwrap()[&key].textures[&(r.as_raw() as usize)].first;
            assert_eq!(run(&again), run(&textures[0]), "the same address and size: the same run");

            // copies into two neighbours, read back (the texture placed again is not read through:
            // placed textures in UNKNOWN layout inherit no contents from one another in D3D12)
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            dev.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
            let buffer = |t: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
                let d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: 3 * total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: t, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut r).unwrap();
                r.unwrap()
            };
            let (upload, readback) = (buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ), buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST));
            let mut p = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut p)).unwrap();
            for k in 0..2usize {
                std::slice::from_raw_parts_mut((p as *mut u8).add(k * total as usize), total as usize).fill(0x11 * (k as u8 + 1));
            }
            upload.Unmap(0, None);
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let loc = |r: &ID3D12Resource, offset: u64| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: offset, ..fp } } };
            let sub = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
            for k in 0..2 {
                list.CopyTextureRegion(&sub(&textures[k]), 0, 0, 0, &loc(&upload, k as u64 * total), None);
            }
            let barrier = |r: &ID3D12Resource| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: D3D12_RESOURCE_STATE_COPY_DEST, StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ResourceBarrier(&[barrier(&textures[0]), barrier(&textures[1])]);
            for (k, t) in [&textures[0], &textures[1]].into_iter().enumerate() {
                list.CopyTextureRegion(&loc(&readback, k as u64 * total), 0, 0, 0, &sub(t), None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            let t = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert_eq!(fence.GetCompletedValue(), 1, "GPU work did not finish");
            readback.Map(0, None, Some(&mut p)).unwrap();
            for (k, want) in [0x11u8, 0x22].into_iter().enumerate() {
                let row = std::slice::from_raw_parts((p as *const u8).add(k * total as usize), 512 * 4);
                assert!(row.iter().all(|&b| b == want), "copy {k} read back {:02x?}", &row[..16]);
            }
            readback.Unmap(0, None);

            // superseded: the run goes back (and comes back while nobody took it); released: gone
            unmap_superseded(&textures[1]);
            assert_eq!(held(&textures[1]), Some(false));
            assert!(remap_if_unmapped(&textures[1]) == Remap::Mapped && held(&textures[1]) == Some(true));
            for t in textures.iter().chain([&c, &again]) {
                forget(t.as_raw() as usize);
                assert_eq!(held(t), None);
            }
        }
    }

    /// Only textures in the layout the driver picks take the block heaps.
    #[test]
    fn textures_in_a_layout_of_their_own_keep_the_pages() {
        let desc = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 512, Height: 512, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, ..Default::default() };
        assert!(takes_blocks(&desc));
        assert!(!takes_blocks(&D3D12_RESOURCE_DESC { Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, ..desc }));
    }

    /// One call per heap: its tiles in runs, the other heaps' skipped.
    #[test]
    fn tile_ranges_per_heap() {
        let tiles = [(0, 5), (0, 6), (1, 0), (1, 1), (0, 9), (0, 10), (0, 11)];
        assert_eq!(heap_ranges(0, &tiles), [(true, 5, 2), (false, 0, 2), (true, 9, 3)]);
        assert_eq!(heap_ranges(1, &tiles), [(false, 0, 2), (true, 0, 2), (false, 0, 3)]);
        // shared pages out of order: runs break where the tiles do
        assert_eq!(heap_ranges(0, &[(0, 3), (0, 1), (0, 2)]), [(true, 3, 1), (true, 1, 2)]);
    }

    /// Idle pages go back after `IDLE`, and a sweep looks at the queue of idle pages alone: one
    /// used again stays, one idle again later goes when that is `IDLE` ago, and the sweeps of
    /// 20 000 placements beside 200 000 pages in use take no time to speak of (a look at every
    /// page at each of them: 4 000 000 000 looks).
    #[test]
    fn idle_pages_are_swept_from_their_queue() {
        use std::collections::{HashMap, VecDeque};
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut idle: VecDeque<(PageKey, Instant)> = VecDeque::new();
        let mut pages: HashMap<usize, Page> = (0..200_000usize).map(|i| (i * PAGE, Page { tile: (0, i as u32), users: 1, idle_since: None })).collect();
        let mut extra: HashMap<(usize, usize), Page> = HashMap::from([((7 * PAGE, 3), Page { tile: (1, 0), users: 1, idle_since: None })]);
        let mut cooling: VecDeque<(Tile, Instant)> = VecDeque::new();
        // what `release_tiles` and `bind` do to a page
        fn release(p: &mut Page, idle: &mut VecDeque<(PageKey, Instant)>, key: PageKey, when: Instant) {
            (p.users, p.idle_since) = (0, Some(when));
            idle.push_back((key, when));
        }
        fn use_again(p: &mut Page) {
            (p.users, p.idle_since) = (1, None);
        }
        // pages 1, 2, 3 idle at 0 ms and an extra page at 100 ms; page 2 is used again at
        // 500 ms; page 3 too, and is idle again at 1500 ms
        for a in [PAGE, 2 * PAGE, 3 * PAGE] {
            release(pages.get_mut(&a).unwrap(), &mut idle, PageKey::Own(a), at(0));
        }
        release(extra.get_mut(&(7 * PAGE, 3)).unwrap(), &mut idle, PageKey::Extra(7 * PAGE, 3), at(100));
        use_again(pages.get_mut(&(2 * PAGE)).unwrap());
        use_again(pages.get_mut(&(3 * PAGE)).unwrap());
        release(pages.get_mut(&(3 * PAGE)).unwrap(), &mut idle, PageKey::Own(3 * PAGE), at(1500));

        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(1900));
        assert!(cooling.is_empty() && pages.len() == 200_000 && idle.len() == 5, "nothing is idle for IDLE yet");
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2050));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1)], "page 1 alone: 2 is in use, 3 idle since 1500 ms");
        assert!(!pages.contains_key(&PAGE) && pages.contains_key(&(2 * PAGE)) && pages.contains_key(&(3 * PAGE)) && extra.len() == 1);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(2150));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0)], "the extra page, idle since 100 ms");
        assert!(extra.is_empty() && pages.len() == 199_999);
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3499));
        assert!(pages.contains_key(&(3 * PAGE)));
        sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(3500));
        assert_eq!(cooling.iter().map(|c| c.0).collect::<Vec<_>>(), [(0, 1), (1, 0), (0, 3)]);
        assert!(idle.is_empty() && pages.len() == 199_998 && pages[&(2 * PAGE)].users == 1);

        // every placement sweeps: with nothing idle that is nothing, whatever is in use
        let t = Instant::now();
        for k in 0..20_000u64 {
            sweep_idle(&mut idle, &mut pages, &mut extra, &mut cooling, at(4000 + k));
        }
        assert!(t.elapsed() < Duration::from_secs(2), "{:?} for 20 000 sweeps", t.elapsed());
        assert_eq!(pages.len(), 199_998);
    }

    /// Any 64 KiB page boundary inside `[va, va + bytes)` belongs to that range alone, so
    /// non-overlapping ranges never share an owned page even when only 4 KiB aligned.
    #[test]
    fn owned_pages_never_alias_disjoint_ranges() {
        assert_eq!(owned_pages(0x1_0000, Some(0x1_0000), 4), Some((0x1_0000, 1)));
        assert_eq!(owned_pages(0x1_0000, Some(0x1_4000), 4), Some((0x1_0000, 2)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4000), 4), Some((0x2_0000, 0)));
        assert_eq!(owned_pages(0x1_4000, Some(0x1_0000), 4), Some((0x2_0000, 1)));
        assert_eq!(owned_pages(0x1_4000, Some(0x4_0000), 4), Some((0x2_0000, 4)));
        assert_eq!(owned_pages(0x1_4000, None, 4), None);
        // Exhaustively check 4 KiB-aligned disjoint pairs over 256 KiB.
        const K4: usize = 0x1000;
        for a in (0..16).map(|i| 0x10_0000 + i * K4) {
            for sa in [K4, 4 * K4, 16 * K4, 20 * K4] {
                let (fa, na) = owned_pages(a, Some(sa as u64), 8).unwrap();
                let pa: Vec<usize> = (0..na).map(|i| fa + i * PAGE).collect();
                for b in (0..16).map(|i| 0x10_0000 + i * K4) {
                    for sb in [K4, 4 * K4, 16 * K4, 20 * K4] {
                        if a + sa <= b || b + sb <= a {
                            let (fb, nb) = owned_pages(b, Some(sb as u64), 8).unwrap();
                            for j in 0..nb {
                                assert!(!pa.contains(&(fb + j * PAGE)), "disjoint [{a:#x}..{:#x}) and [{b:#x}..{:#x}) shared page", a + sa, b + sb);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Reserved only with TILED support, and for MSAA a tiled sample count (#940: the driver
    /// refused R32G8X24_TYPELESS reserved, at 1 and 4 samples, and the debug layer said so).
    #[test]
    fn reserved_needs_tiled_format_and_tiled_msaa() {
        let tiled = D3D12_FORMAT_SUPPORT2_TILED.0;
        assert!(can_reserve(tiled, 1, 0));
        assert!(!can_reserve(0, 1, 0), "format without TILED");
        assert!(can_reserve(tiled, 4, 1));
        assert!(!can_reserve(tiled, 4, 0), "no tiled quality level at 4 samples");
        assert!(!can_reserve(0, 4, 1));
    }

    /// The device's answer agrees with CreateReservedResource: what `reservable` turns away the
    /// driver refuses, what it lets through the driver makes (WARP or a hardware adapter).
    #[test]
    fn reservable_agrees_with_the_driver() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: windows::Win32::Graphics::Dxgi::IDXGIFactory4 = windows::Win32::Graphics::Dxgi::CreateDXGIFactory1().expect("DXGI factory");
                let Ok(warp) = factory.EnumWarpAdapter::<windows::Win32::Graphics::Dxgi::IDXGIAdapter>() else { return };
                let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
            }
            let Some(dev) = dev else { return };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &mut o as *mut _ as *mut c_void, std::mem::size_of_val(&o) as u32).is_err() || o.TiledResourcesTier.0 < D3D12_TILED_RESOURCES_TIER_2.0 {
                return;
            }
            for (format, samples) in [(DXGI_FORMAT_R32G8X24_TYPELESS, 1), (DXGI_FORMAT_R32G8X24_TYPELESS, 4), (DXGI_FORMAT_R32_TYPELESS, 1), (DXGI_FORMAT_R8G8B8A8_UNORM, 1)] {
                let flags = if format == DXGI_FORMAT_R8G8B8A8_UNORM { D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET } else { D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: 128, Height: 128, DepthOrArraySize: 1, MipLevels: 1, Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: samples, Quality: 0 }, Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE, Flags: flags, ..Default::default()
                };
                let mut r: Option<ID3D12Resource> = None;
                let made = dev.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r).is_ok() && r.is_some();
                assert_eq!(reservable(&dev, &desc), made, "format {} x{samples}: reservable vs CreateReservedResource", format.0);
            }
        }
    }

    /// A placement's arena id, with heaps numbered as they are made.
    fn take(a: &mut Arenas<u32>, va: usize, bytes: usize, align: usize, now: Instant) -> u64 {
        let n = a.list.len() as u32;
        a.take(va, bytes, align, now, |_, _| Some(n)).unwrap().2
    }

    #[test]
    fn an_arena_is_shared_at_its_addresses_and_otherwise_its_placements_size() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let first = take(&mut a, a0, 70 * MIB, PAGE, t);
        assert_eq!(a.bytes(), 70 * MIB, "the placement's size, not 256 MiB");
        // the same address, and a smaller placement inside: the same memory
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t), first);
        assert_eq!(take(&mut a, a0 + 4 * MIB, 8 * MIB, PAGE, t), first);
        // one straddling its end gets an arena of its own size; so does an MSAA target that needs 4 MiB
        let straddle = take(&mut a, a0 + 64 * MIB, 70 * MIB, PAGE, t);
        let msaa = take(&mut a, a0 + 8 * MIB, 8 * MIB, 4 * MIB, t);
        assert!(straddle != first && msaa != first && msaa != straddle);
        assert_eq!(a.bytes(), (70 + 70 + 8) * MIB);
        // the newest arena holding an address is the one placements there share
        assert_eq!(take(&mut a, a0 + 64 * MIB, 4 * MIB, PAGE, t), straddle);
    }

    #[test]
    fn an_arena_no_resource_uses_goes_after_two_seconds() {
        let (t, a0) = (Instant::now(), 0x1_0000_0000);
        let mut a = Arenas::new();
        let id = take(&mut a, a0, 70 * MIB, PAGE, t);
        a.release(id, t);
        // placed again within 2 s: the same memory (with what it holds)
        assert_eq!(take(&mut a, a0, 70 * MIB, PAGE, t + Duration::from_secs(1)), id);
        a.release(id, t + Duration::from_secs(1));
        take(&mut a, a0 + 512 * MIB, MIB, PAGE, t + Duration::from_secs(3) + Duration::from_millis(1));
        assert_eq!(a.bytes(), MIB, "the idle arena went");
    }

    /// The retail game's pattern (HANDOFF): targets packed back to back over 3.5 GB of its
    /// addresses and placed again and again, each round at the last round's addresses or a third
    /// of a target further on, so placements straddle earlier ones. Main's 256 MiB arenas merged
    /// straddled arenas into growing unions while the old ones lived on (15 GB in 2 s).
    #[test]
    fn streamed_targets_stay_bounded() {
        const S: usize = 64 * MIB; // a 3840x2160 D32S8, rounded
        const K: usize = 8; // targets a round
        const SPAN: usize = 3584 * MIB;
        let (t0, a0) = (Instant::now(), 0x1_0000_0000usize);
        let mut a = Arenas::new();
        let mut live: std::collections::VecDeque<Vec<u64>> = std::collections::VecDeque::new();
        let mut bytes = Vec::new();
        for r in 0..400usize {
            let now = t0 + Duration::from_millis(500 * r as u64);
            let start = ((r / 2) * S / 3) % (SPAN - K * S) & !(PAGE - 1);
            live.push_back((0..K).map(|k| take(&mut a, a0 + start + k * S, S, PAGE, now)).collect());
            // a target lives three rounds (1.5 s)
            if live.len() > 3 {
                for id in live.pop_front().unwrap() {
                    a.release(id, now);
                }
            }
            bytes.push(a.bytes());
        }
        // live: 3 rounds of targets; idle and kept (< 2 s): 4 more
        let most = *bytes.iter().max().unwrap();
        assert!(most <= 7 * K * S, "{most} bytes of arenas");
        let early = *bytes[20..120].iter().max().unwrap();
        let late = *bytes[300..].iter().max().unwrap();
        assert!(late <= early, "arenas grow over time: {early} then {late} bytes");
    }

    /// On the GPU: D32S8 targets 256 MiB apart take their own size of video memory, not an arena
    /// of 256 MiB each; depth and stencil survive a clear, CopyResource and a read of both planes;
    /// arenas no target uses go. Needs a hardware adapter (skips on WARP).
    #[test]
    fn d32s8_targets_take_their_own_size_of_video_memory() {
        let _vram_guard = VRAM_TEST_LOCK.lock().unwrap();
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::Graphics::Dxgi::*;
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let Ok(adapter) = factory.EnumAdapters1(0) else { return };
            if adapter.GetDesc1().is_ok_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                return;
            }
            let mut dev: Option<ID3D12Device> = None;
            let _ = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev);
            let Some(dev) = dev.filter(|d| tier2(d)) else { return };
            super::super::real::adopt(dev.as_raw(), super::super::real::Kind::Device);
            let memory: IDXGIAdapter3 = adapter.cast().unwrap();
            let local = || {
                let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                let _ = memory.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i);
                i.CurrentUsage
            };
            let sample = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
            let tex = |w: u64, h: u32| D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1, MipLevels: 1, Format: DXGI_FORMAT_R32G8X24_TYPELESS, SampleDesc: sample, Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default() };
            let ds = D3D12_DEPTH_STENCIL_VALUE { Depth: 1.0, Stencil: 0 };
            let clear = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, Anonymous: D3D12_CLEAR_VALUE_0 { DepthStencil: ds } };
            let riid = &ID3D12Resource::IID as *const windows::core::GUID as *const windows_sys::core::GUID;
            let place_at = |va: usize, d: &D3D12_RESOURCE_DESC| {
                let mut p = std::ptr::null_mut();
                let hr = place(dev.as_raw(), va, d, D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32, &clear as *const _ as *const c_void, None, riid, &mut p);
                assert!(hr.is_some_and(|hr| hr >= 0) && !p.is_null(), "D32S8 target at {va:#x}");
                ID3D12Resource::from_raw(p)
            };

            // 14 4K targets over 3.5 GB of addresses
            let big = tex(3840, 2160);
            let size = dev.GetResourceAllocationInfo(0, &[big]).SizeInBytes as usize;
            let (used0, arenas0) = (local(), arena_bytes().0);
            let targets: Vec<ID3D12Resource> = (0..14).map(|k| place_at(0x40_0000_0000 + k * 256 * MIB, &big)).collect();
            let (used, arenas) = (local().saturating_sub(used0) as usize, arena_bytes().0 - arenas0);
            println!("14 D32S8 3840x2160 targets ({size} bytes each): {arenas} bytes of arenas, {used} bytes more local video memory");
            assert!(arenas <= 14 * size.next_multiple_of(PAGE), "{arenas} bytes of arenas");
            assert!(used < 14 * 256 * MIB, "{used} bytes of video memory");
            drop(targets);

            // clear one, copy it to another, read both planes back
            let small = tex(64, 64);
            let (a, b) = (place_at(0x48_0000_0000, &small), place_at(0x48_1000_0000, &small));
            let dsv_heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV, NumDescriptors: 1, ..Default::default() }).unwrap();
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            let view = D3D12_DEPTH_STENCIL_VIEW_DESC { Format: DXGI_FORMAT_D32_FLOAT_S8X24_UINT, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D, ..Default::default() };
            dev.CreateDepthStencilView(&a, Some(&view), dsv);
            let mut layouts = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
            let mut total = 0u64;
            dev.GetCopyableFootprints(&small, 0, 2, 0, Some(layouts.as_mut_ptr()), None, None, Some(&mut total));
            let buffer = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1, SampleDesc: sample, Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ..Default::default() };
            let mut readback: Option<ID3D12Resource> = None;
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).unwrap();
            let readback = readback.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let barrier = |r: &ID3D12Resource, from: D3D12_RESOURCE_STATES, to: D3D12_RESOURCE_STATES| {
                let t = D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, StateBefore: from, StateAfter: to };
                D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, Anonymous: D3D12_RESOURCE_BARRIER_0 { Transition: std::mem::ManuallyDrop::new(t) }, ..Default::default() }
            };
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25, 0x5A, None);
            list.ResourceBarrier(&[barrier(&a, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(&b, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.CopyResource(&b, &a);
            list.ResourceBarrier(&[barrier(&b, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for (plane, layout) in layouts.iter().enumerate() {
                let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&readback), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *layout } };
                let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&b), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: plane as u32 } };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            // no event: returns when the fence has the value
            fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).unwrap();
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let bytes = std::slice::from_raw_parts(p as *const u8, total as usize);
            let row = |plane: usize, y: usize| layouts[plane].Offset as usize + y * layouts[plane].Footprint.RowPitch as usize;
            for y in [0, 31, 63] {
                let depth = f32::from_le_bytes(bytes[row(0, y)..row(0, y) + 4].try_into().unwrap());
                assert_eq!((depth, bytes[row(1, y)]), (0.25, 0x5A), "row {y}: depth, stencil");
            }
            readback.Unmap(0, None);
            drop((a, b));

            // 2 s later the arenas no target uses are let go: one new target's is all that is left
            std::thread::sleep(IDLE + Duration::from_millis(100));
            let last = place_at(0x50_0000_0000, &small);
            let small_size = dev.GetResourceAllocationInfo(0, &[small]).SizeInBytes as usize;
            assert_eq!(arena_bytes().0 - arenas0, small_size.next_multiple_of(PAGE));
            drop(last);
        }
    }
}
