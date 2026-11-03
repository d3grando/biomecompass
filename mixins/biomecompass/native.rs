// BiomeCompass's adapter to the game (Horizon aa27eca326db26e999d4e59d7899576016764b1e,
// see horizon-mixin.toml). Injected as `crate::game::biomecompass_native`; it binds
// the game-independent search in `horizon_client::biomecompass` (runtime.rs) and
// the Biome Compass item (`horizon_client::biomecompass_item`, item.rs) to the
// game that simulates the world (in-frame single player, single player's server
// thread, a LAN host, a dedicated server), and draws the needle in the window.
//
// The Biome Compass is the server's: a right-click (`use_potion_item`'s HEAD,
// the first item use a compass reaches) opens the picker, a vanilla dialog the
// server sends; its buttons run `/biomecompass` as the player; every few ticks
// the server sets the compasses in each player's inventory to their search
// (vanilla's lodestone tracker), so any client draws the needle.
//
// Biomes come from the world generator of the level in place
// (`end.locator()`, the `VanillaTerrain` behind `/locate biome`), sampled with
// `noise_biomes`: `BiomeSource.getNoiseBiome` at quart positions, which needs
// no chunk to be generated or loaded. A search keeps that generator, so a
// player who changes dimension mid-search doesn't change what it searches.
use super::Game;
use horizon_client::biomecompass::{
    self as bc, Actor, Click, Command, Config, Entry, Heights, Needle, Phase, Position, Resolved,
    Search, Spec, Teleport, Track, Viewer,
};
use horizon_client::biomecompass_item as item;
use horizon_client::extras::Corner;
use horizon_client::extras_hud::Corners;
use horizon_client::hud::{HudBatch, rgba};
use horizon_client::items::ItemStack;
use horizon_client::lan::playerdata::uuid_ints;
use horizon_client::streaming::ChunkGenerator;
use horizon_client::text::{self, ClickEvent, Style, Text};
use horizon_client::vanillagen::VanillaTerrain;
use horizon_protocol::packets::play::{self, Clientbound as C};
use horizon_world::nbt::Tag;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the window and the game simulating the world share (single
/// player's server thread runs in the window's process): the settings, and
/// each player's search as the HUD shows it.
struct Shared {
    config: Option<Config>,
    /// The world the tracks belong to (its folder).
    world: Option<PathBuf>,
    tracks: BTreeMap<Actor, Track>,
}
static SHARED: Mutex<Shared> = Mutex::new(Shared {
    config: None,
    world: None,
    tracks: BTreeMap::new(),
});

/// Options > Mods' on/off as the window last saw it: single player's
/// server thread keeps the options it started with, so it follows the
/// window's when there is one.
static WINDOW: AtomicU8 = AtomicU8::new(UNSEEN);
const UNSEEN: u8 = 0;
const WINDOW_ON: u8 = 1;
const WINDOW_OFF: u8 = 2;

/// A running search and the generator it samples.
struct Active {
    search: Search,
    terrain: Arc<VanillaTerrain>,
    ticks: u32,
}

/// The searches of one world, on the thread of the game that simulates it.
struct Engine {
    world: PathBuf,
    searches: BTreeMap<Actor, Active>,
    /// Ticks left showing a search that found nothing.
    missing: BTreeMap<Actor, u32>,
    /// Pickers to send at the next tick (a right-click happens in the
    /// guest's turn, while the hub that reaches them is out of the game).
    pickers: Vec<(Actor, Tag)>,
    /// Ticks since the engine was made.
    clock: u32,
    /// When each player last opened the picker with a right-click.
    used: BTreeMap<Actor, u32>,
}

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    /// When the window's HUD first drew a turning needle.
    static SPIN: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
}

/// Never held while the game runs.
fn shared<T>(run: impl FnOnce(&mut Shared) -> T) -> T {
    let mut s = SHARED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run(&mut s)
}

/// The settings file as it is now (the problems logged); a missing one is
/// written with the defaults, so there is a file to edit by hand.
fn read_config(game: &Game<'_>) -> (Config, Vec<String>) {
    match game.native_setting(bc::ID, bc::CONFIG_FILE) {
        Some(text) => {
            let (config, problems) = Config::parse(&text);
            for p in &problems {
                log::warn!("BiomeCompass: {}: {p}", bc::CONFIG_FILE);
            }
            (config, problems)
        }
        None => {
            let config = Config::default();
            game.native_set_setting(bc::ID, bc::CONFIG_FILE, &config.to_file());
            (config, Vec::new())
        }
    }
}

/// The settings in effect, read once per process.
fn config(game: &Game<'_>) -> Config {
    if let Some(c) = shared(|s| s.config.clone()) {
        return c;
    }
    let (read, _) = read_config(game);
    shared(|s| s.config.get_or_insert(read).clone())
}

/// Puts `config` in effect everywhere.
fn publish(config: Config) {
    shared(|s| s.config = Some(config));
}

/// BiomeCompass isn't turned off in Options > Mods. Every hook but the
/// list and the settings page does nothing when it is (until the
/// launcher's next build leaves it out).
fn on(game: &Game<'_>) -> bool {
    let own = game.native_on(bc::ID);
    let on = if game.native_window() {
        WINDOW.store(if own { WINDOW_ON } else { WINDOW_OFF }, Ordering::Relaxed);
        own
    } else {
        match WINDOW.load(Ordering::Relaxed) {
            UNSEEN => own,
            seen => seen == WINDOW_ON,
        }
    };
    // For the hooks in the game's library (crafting menu, command tree).
    bc::set_enabled(on);
    on
}

fn corner(c: bc::Corner) -> Corner {
    match c {
        bc::Corner::TopLeft => Corner::TopLeft,
        bc::Corner::TopRight => Corner::TopRight,
        bc::Corner::BottomLeft => Corner::BottomLeft,
        bc::Corner::BottomRight => Corner::BottomRight,
    }
}

/// A biome's name in the game's language, else one made from its id.
fn biome_name(id: &str) -> String {
    let key = bc::translation_key(id);
    let shown = Text::translate(&key, &[]).plain();
    if shown.is_empty() || shown == key {
        bc::pretty(id)
    } else {
        shown
    }
}

/// Runs `f` with this world's engine, made on first use; the engine is out
/// of its slot meanwhile.
fn with_engine<T>(game: &mut Game<'_>, f: impl FnOnce(&mut Game<'_>, &mut Engine) -> T) -> T {
    let kept = ENGINE.with(|e| e.borrow_mut().take());
    let mut engine = match kept {
        Some(e) if e.world == game.world_root => e,
        _ => Engine {
            world: game.world_root.clone(),
            searches: BTreeMap::new(),
            missing: BTreeMap::new(),
            pickers: Vec::new(),
            clock: 0,
            used: BTreeMap::new(),
        },
    };
    let out = f(game, &mut engine);
    ENGINE.with(|e| *e.borrow_mut() = Some(engine));
    out
}

/// A line for one player, by UUID, outside their command: the window's
/// own player in-frame, else the hub's guest (single player's own window
/// is one of its server thread). A player who left gets nothing.
fn tell(game: &mut Game<'_>, actor: Actor, line: Text) {
    let local = game.native_window()
        && game.remote.is_none()
        && game.lan_acting.is_none()
        && game.player.is_some()
        && game.level.player_uuid == Some(actor);
    if local {
        game.system_message(line);
        return;
    }
    let Some(host) = game.lan.as_mut() else {
        return;
    };
    let to = host
        .hub
        .guests
        .iter()
        .find(|g| g.placed && horizon_client::lan::playerdata::uuid_ints(g.uuid) == actor);
    if let Some(conn) = to.map(|g| g.conn) {
        host.hub.send_to(
            conn,
            &C::SystemChat(play::SystemChat {
                content: horizon_client::lan::data::component_nbt(&line.component()),
                overlay: false,
            }),
        );
    }
}

/// Clickable coordinates: a click puts `/biomecompass tp` in the chat box.
fn coordinates(pos: Position) -> Text {
    Text::literal(format!("[{}, {}, {}]", pos[0], pos[1], pos[2])).with_style(&Style {
        click: Some(ClickEvent::SuggestCommand(format!("/{} tp", bc::COMMAND))),
        hover: Some(Box::new(Text::literal("Teleport there (operators)"))),
        ..Style::color(text::GREEN)
    })
}

/// A biome as a link that starts a search for it.
fn biome_link(entry: &Entry) -> Text {
    Text::literal(entry.name.clone()).with_style(&Style {
        click: Some(ClickEvent::RunCommand(format!(
            "/{} find {}",
            bc::COMMAND,
            entry.id
        ))),
        hover: Some(Box::new(Text::literal(format!(
            "{}\nClick to search for it",
            entry.id
        )))),
        ..Style::color(text::GREEN)
    })
}

impl Game<'_> {
    /// The player acting now, as BiomeCompass knows them: UUID, block
    /// position and dimension. None without one (a server's console). A
    /// guest's level is in place while they act.
    fn bc_player(&mut self) -> Option<(Actor, Position, &'static str)> {
        let p = self.player.as_ref()?;
        let feet = [
            p.position.x.floor() as i32,
            p.position.y.floor() as i32,
            p.position.z.floor() as i32,
        ];
        if self.lan_acting.is_none() && !self.native_window() {
            return None;
        }
        Some((self.acting_player_uuid(), feet, self.dim.id()))
    }

    /// The window's own player, as the game simulating the world keys
    /// them: the profile it joined with, else the world's player.
    fn bc_window_actor(&self) -> Actor {
        match &self.remote {
            Some(remote) => horizon_client::lan::playerdata::uuid_ints(remote.profile()),
            None => self.level.player_uuid.unwrap_or_default(),
        }
    }

    /// The biomes of a biome tag (nested tags expanded): the data pack's
    /// files, else the tags the game loaded with its mob data.
    fn bc_tag(&self, tag: &str) -> BTreeSet<String> {
        let mut out: BTreeSet<String> = self
            .tag_entries("worldgen/biome", tag)
            .into_iter()
            .collect();
        if out.is_empty() {
            let short = tag.strip_prefix("minecraft:").unwrap_or(tag);
            let data = self.res.mob_data();
            out = self
                .res
                .biomes()
                .list
                .iter()
                .filter(|b| data.biome_has(tag, &b.name) || data.biome_has(short, &b.name))
                .map(|b| b.name.clone())
                .collect();
        }
        out
    }

    /// Every biome the game knows.
    fn bc_all_biomes(&self) -> BTreeSet<String> {
        self.res
            .biomes()
            .list
            .iter()
            .map(|b| b.name.clone())
            .filter(|n| !n.is_empty())
            .collect()
    }

    /// The biomes of the dimension in place, with their names; all biomes
    /// when the dimension has no tag listing them.
    fn bc_entries(&self) -> Vec<Entry> {
        let listed = bc::dimension_tag(self.dim.id()).map(|t| self.bc_tag(t));
        let ids = match listed {
            Some(ids) if !ids.is_empty() => ids,
            _ => self.bc_all_biomes(),
        };
        ids.into_iter()
            .map(|id| Entry {
                name: biome_name(&id),
                id,
            })
            .collect()
    }

    /// `native_command` (a hook point): BiomeCompass's commands, as the
    /// player who typed them; false leaves the line to the game. A window
    /// joined to a server (single player's own server thread included)
    /// passes it on: the server runs it.
    pub(super) fn biomecompass_command(&mut self, command: &str) -> bool {
        if self.remote.is_some() || !on(self) {
            return false;
        }
        let Some(parsed) = bc::parse_command(command) else {
            return false;
        };
        let command = match parsed {
            Ok(c) => c,
            Err(usage) => {
                self.command_failure(Text::literal(usage));
                return true;
            }
        };
        match command {
            Command::Help => {
                for line in bc::help() {
                    self.system_message(Text::literal(line));
                }
            }
            Command::Reload => {
                if self.ui.registries.permission < bc::OPERATOR {
                    self.command_failure(Text::literal(format!(
                        "BiomeCompass reload needs permission level {}",
                        bc::OPERATOR
                    )));
                    return true;
                }
                let (config, problems) = read_config(self);
                publish(config);
                let line = if problems.is_empty() {
                    format!("BiomeCompass: {} read again", bc::CONFIG_FILE)
                } else {
                    format!(
                        "BiomeCompass: {} read again; skipped: {}",
                        bc::CONFIG_FILE,
                        problems.join("; ")
                    )
                };
                self.system_message(Text::literal(line));
            }
            Command::List(filter) => self.bc_list(filter.as_deref()),
            Command::Find(query) => self.bc_find(&query),
            Command::Status => self.bc_status(),
            Command::Stop => self.bc_stop(),
            Command::Teleport => self.bc_teleport(),
            Command::Pick(filter) => self.bc_pick(filter.as_deref()),
            Command::Give => self.bc_give(),
        }
        true
    }

    /// `/biomecompass list [filter]`: the dimension's biomes as links.
    fn bc_list(&mut self, filter: Option<&str>) {
        let entries = bc::filter_entries(&self.bc_entries(), filter);
        let place = bc::dimension_name(self.dim.id());
        if entries.is_empty() {
            self.command_failure(Text::literal(format!(
                "No biome of {place} matches \"{}\"",
                filter.unwrap_or_default()
            )));
            return;
        }
        let mut line = Text::literal(format!(
            "Biomes of {place} ({}), click one to search for it: ",
            entries.len()
        ));
        for (i, e) in entries.iter().enumerate() {
            if i > 0 {
                line.push(Text::literal(", "));
            }
            line.push(biome_link(e));
        }
        self.system_message(line);
    }

    /// `/biomecompass find <biome or #tag>`: a new search around the player
    /// (it replaces theirs, if any).
    fn bc_find(&mut self, query: &str) {
        let Some((actor, origin, dimension)) = self.bc_player() else {
            self.command_failure(Text::literal(
                "BiomeCompass searches around a player in a world",
            ));
            return;
        };
        let Some(terrain) = self.end.locator() else {
            self.command_failure(Text::literal(
                "BiomeCompass: this world has no world generator to search (a superflat or test world)",
            ));
            return;
        };
        let entries = self.bc_entries();
        let here: BTreeSet<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        let (wanted, label) = match bc::resolve(query, &entries) {
            Resolved::Biome(id) => {
                let name = biome_name(&id);
                (BTreeSet::from([id]), name)
            }
            Resolved::Tag(tag) => {
                let all = self.bc_tag(&tag);
                if all.is_empty() {
                    self.command_failure(Text::literal(format!("Unknown biome tag: #{tag}")));
                    return;
                }
                if !all.iter().any(|b| here.contains(b.as_str())) {
                    self.command_failure(Text::literal(format!(
                        "No biome of #{tag} generates in {}",
                        bc::dimension_name(dimension)
                    )));
                    return;
                }
                (all, format!("#{tag}"))
            }
            Resolved::Several(ids) => {
                let mut line = Text::literal(format!("Several biomes match \"{query}\": "));
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        line.push(Text::literal(", "));
                    }
                    if let Some(e) = entries.iter().find(|e| &e.id == id) {
                        line.push(biome_link(e));
                    }
                }
                self.system_message(line);
                return;
            }
            Resolved::Unknown => {
                let id = bc::full_id(query);
                let message = if self.bc_all_biomes().contains(&id) {
                    format!(
                        "{} doesn't generate in {}",
                        biome_name(&id),
                        bc::dimension_name(dimension)
                    )
                } else {
                    format!("Unknown biome: {query} (see /{} list)", bc::COMMAND)
                };
                self.command_failure(Text::literal(message));
                return;
            }
        };
        let config = config(self);
        let layers = bc::layers(
            dimension,
            origin[1],
            self.dim.shape().min_y,
            (self.dim.shape().max_y() - 1),
            config.heights,
        );
        let search = Search::new(Spec {
            wanted,
            label: label.clone(),
            dimension: dimension.to_owned(),
            origin,
            radius: config.radius,
            spacing: config.spacing,
            layers,
        });
        with_engine(self, |_, engine| {
            engine.missing.remove(&actor);
            engine.searches.insert(
                actor,
                Active {
                    search,
                    terrain,
                    ticks: 0,
                },
            );
        });
        shared(|s| {
            s.tracks.insert(
                actor,
                Track {
                    label: label.clone(),
                    dimension: dimension.to_owned(),
                    phase: Phase::Searching {
                        progress: 0.0,
                        radius: config.radius,
                    },
                },
            );
        });
        let mut line = bc::started_message(&label, config.radius, dimension);
        if config.heights == Heights::All && dimension == "minecraft:overworld" {
            line.push_str(" (cave biomes too)");
        }
        self.system_message(Text::literal(line));
    }

    /// `/biomecompass status`.
    fn bc_status(&mut self) {
        let Some((actor, feet, dimension)) = self.bc_player() else {
            self.system_message(Text::literal(bc::status_message(None, [0.0, 0.0], "")));
            return;
        };
        let track = shared(|s| s.tracks.get(&actor).cloned());
        let from = [f64::from(feet[0]) + 0.5, f64::from(feet[2]) + 0.5];
        self.system_message(Text::literal(bc::status_message(
            track.as_ref(),
            from,
            dimension,
        )));
    }

    /// `/biomecompass stop`: no search, no needle.
    fn bc_stop(&mut self) {
        let Some((actor, ..)) = self.bc_player() else {
            return;
        };
        with_engine(self, |_, engine| {
            engine.searches.remove(&actor);
            engine.missing.remove(&actor);
        });
        let had = shared(|s| s.tracks.remove(&actor)).is_some();
        let line = if had {
            "BiomeCompass: stopped; the needle is put away"
        } else {
            "BiomeCompass: nothing to stop"
        };
        self.system_message(Text::literal(line));
    }

    /// The height to stand at on column (x, z): on the highest block when
    /// its chunk is loaded, else on the generator's terrain (above water).
    fn bc_landing(&self, terrain: &VanillaTerrain, x: i32, z: i32) -> Option<i32> {
        if self.world.loaded(x, z) {
            let (lo, hi) = (self.dim.shape().min_y, (self.dim.shape().max_y() - 1));
            return (lo..=hi)
                .rev()
                .find(|&y| !self.world.is_air(self.world.block(x, y, z)))
                .map(|y| y + 1);
        }
        let mut worker = terrain.worker();
        let surface = terrain.lod_surface(&mut worker, x, z)?;
        Some(
            surface
                .water
                .map_or(surface.height, |(top, _)| top.max(surface.height)),
        )
    }

    /// `/biomecompass tp`: to the biome found, through the game's own `/tp`
    /// (so its permission check and its way of moving any player apply).
    fn bc_teleport(&mut self) {
        let config = config(self);
        if config.teleport == Teleport::Off {
            self.command_failure(Text::literal(
                "BiomeCompass: teleporting is turned off in its settings",
            ));
            return;
        }
        if self.ui.registries.permission < bc::OPERATOR {
            self.command_failure(Text::literal(format!(
                "BiomeCompass: teleporting needs permission level {} (operators)",
                bc::OPERATOR
            )));
            return;
        }
        let Some((actor, _, dimension)) = self.bc_player() else {
            self.command_failure(Text::literal("BiomeCompass teleports a player in a world"));
            return;
        };
        let track = shared(|s| s.tracks.get(&actor).cloned());
        let Some(Track {
            dimension: there,
            phase: Phase::Found { pos, .. },
            ..
        }) = track
        else {
            self.command_failure(Text::literal(
                "BiomeCompass: nothing found yet to teleport to",
            ));
            return;
        };
        if there != dimension {
            self.command_failure(Text::literal(format!(
                "BiomeCompass: the biome is in {}",
                bc::dimension_name(&there)
            )));
            return;
        }
        if dimension == "minecraft:the_nether" {
            self.command_failure(Text::literal(
                "BiomeCompass doesn't teleport in the Nether (no safe height to land at); follow the needle",
            ));
            return;
        }
        let Some(terrain) = self.end.locator() else {
            return;
        };
        let Some(y) = self.bc_landing(&terrain, pos[0], pos[2]) else {
            self.command_failure(Text::literal(
                "BiomeCompass: no ground there to land on; follow the needle",
            ));
            return;
        };
        let (x, z) = (f64::from(pos[0]) + 0.5, f64::from(pos[2]) + 0.5);
        self.run_command(&format!("tp @s {x:.1} {y} {z:.1}"));
    }

    /// `native_tick` (a hook point): a slice of every search, then the
    /// lines and pickers for players, and every few ticks their Biome
    /// Compasses. The window's game notes Options > Mods' on/off for single
    /// player's server thread, and draws only (unless in-frame).
    pub(super) fn biomecompass_tick(&mut self) {
        bc::set_client_thread(self.bc_foreign_client());
        if !on(self) || self.remote.is_some() {
            return;
        }
        // A world opened anew starts with no needles.
        let root = self.world_root.clone();
        shared(|s| {
            if s.world.as_ref() != Some(&root) {
                s.world = Some(root);
                s.tracks.clear();
            }
        });
        let config = config(self);
        let (lines, pickers, refresh) = with_engine(self, |_, engine| {
            engine.clock = engine.clock.wrapping_add(1);
            let refresh = engine.clock % bc::COMPASS_TICKS == 0;
            let pickers = std::mem::take(&mut engine.pickers);
            let mut lines: Vec<(Actor, Text)> = Vec::new();
            engine.missing.retain(|actor, left| {
                *left = left.saturating_sub(1);
                if *left == 0 {
                    shared(|s| {
                        let shown = s.tracks.get(actor);
                        if shown.is_some_and(|t| matches!(t.phase, Phase::Missing { .. })) {
                            s.tracks.remove(actor);
                        }
                    });
                }
                *left > 0
            });
            let deadline = Instant::now() + Duration::from_millis(bc::TICK_TIME_MS);
            let count = engine.searches.len().max(1);
            let share = (config.samples_per_tick as usize / count).max(1);
            let mut done = Vec::new();
            for (&actor, active) in engine.searches.iter_mut() {
                let mut used = 0;
                while used < share && !active.search.finished() && Instant::now() < deadline {
                    let columns = active.search.next_columns((share - used).min(bc::BATCH));
                    if columns.is_empty() {
                        break;
                    }
                    let quarts: Vec<[i32; 3]> = active
                        .search
                        .probes(&columns)
                        .into_iter()
                        .map(|p| [p[0] >> 2, p[1] >> 2, p[2] >> 2])
                        .collect();
                    used += quarts.len();
                    let biomes = active.terrain.noise_biomes(&quarts);
                    active.search.feed(&columns, &biomes);
                }
                active.ticks += 1;
                let spec = &active.search.spec;
                if !active.search.finished() {
                    shared(|s| {
                        if let Some(t) = s.tracks.get_mut(&actor) {
                            t.phase = Phase::Searching {
                                progress: active.search.progress(),
                                radius: spec.radius,
                            };
                        }
                    });
                    if active.ticks % bc::PROGRESS_TICKS == 0 {
                        let line = bc::progress_message(
                            &spec.label,
                            active.search.searched(),
                            spec.radius,
                        );
                        lines.push((actor, Text::literal(line)));
                    }
                    continue;
                }
                done.push(actor);
                let phase = match active.search.best() {
                    Some(hit) => {
                        let name = biome_name(&hit.biome);
                        let from = [
                            f64::from(spec.origin[0]) + 0.5,
                            f64::from(spec.origin[2]) + 0.5,
                        ];
                        let (before, after) = bc::found_message(&name, &spec.label, hit.pos, from);
                        let line = Text::literal(before)
                            .append(coordinates(hit.pos))
                            .append(Text::literal(after));
                        lines.push((actor, line));
                        Phase::Found { name, pos: hit.pos }
                    }
                    None => {
                        let line = bc::missing_message(&spec.label, spec.radius);
                        lines.push((actor, Text::literal(line).with_color(text::RED)));
                        engine_missing(&mut engine.missing, actor);
                        Phase::Missing {
                            radius: spec.radius,
                        }
                    }
                };
                shared(|s| {
                    if let Some(t) = s.tracks.get_mut(&actor) {
                        t.phase = phase;
                    }
                });
            }
            for actor in done {
                engine.searches.remove(&actor);
            }
            (lines, pickers, refresh)
        });
        for (actor, line) in lines {
            tell(self, actor, line);
        }
        for (actor, dialog) in pickers {
            self.bc_send_picker(actor, dialog);
        }
        if refresh {
            self.bc_compasses(&config);
        }
    }

    /// Every few ticks: each player's Biome Compasses point at their search
    /// (vanilla's lodestone tracker, which every client draws; their lore
    /// names it), and a player holding one sees it over the hotbar. The hub
    /// sends a guest the stacks that changed, as for any change of their
    /// inventory.
    fn bc_compasses(&mut self, config: &Config) {
        if self.lan_acting.is_some() {
            return;
        }
        let tracks = shared(|s| s.tracks.clone());
        // The window's own player, in-frame (the tick returned already when
        // the window is a client).
        if self.native_window() && self.player.is_some() {
            let me = self.bc_window_actor();
            let pointing = bc::pointing(tracks.get(&me));
            let selected = self.selected;
            let mut held = false;
            for (i, slot) in self.inventory.iter_mut().enumerate() {
                if let Some(s) = slot.as_mut().filter(|s| item::is_biome_compass(s)) {
                    item::point(s, &pointing);
                    held |= i == selected;
                }
            }
            if let Some(s) = self.offhand.as_mut().filter(|s| item::is_biome_compass(s)) {
                item::point(s, &pointing);
                held = true;
            }
            if held
                && config.action_bar
                && let (Some(track), Some(p)) = (tracks.get(&me), self.native_player())
            {
                let viewer = Viewer {
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    yaw: p.yaw,
                    dimension: p.dimension.to_owned(),
                };
                let (line, rgb) = bc::action_bar(track, &viewer);
                self.titles
                    .set_overlay(Text::literal(line).with_color(rgb), false);
            }
        }
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        let mut bars = Vec::new();
        for g in host.hub.guests.iter_mut().filter(|g| g.placed) {
            let actor = uuid_ints(g.uuid);
            let pointing = bc::pointing(tracks.get(&actor));
            let mut held = false;
            for (i, slot) in g.inventory.iter_mut().enumerate() {
                if let Some(s) = slot.as_mut().filter(|s| item::is_biome_compass(s)) {
                    item::point(s, &pointing);
                    held |= i == g.selected;
                }
            }
            if let Some(s) = g.offhand.as_mut().filter(|s| item::is_biome_compass(s)) {
                item::point(s, &pointing);
                held = true;
            }
            if held
                && config.action_bar
                && let Some(track) = tracks.get(&actor)
            {
                let viewer = Viewer {
                    x: g.position[0],
                    y: g.position[1],
                    z: g.position[2],
                    yaw: g.yaw,
                    dimension: g.dimension.clone(),
                };
                let (line, rgb) = bc::action_bar(track, &viewer);
                bars.push((g.conn, Text::literal(line).with_color(rgb)));
            }
        }
        for (conn, bar) in bars {
            host.hub.send_to(
                conn,
                &C::SystemChat(play::SystemChat {
                    content: horizon_client::lan::data::component_nbt(&bar.component()),
                    overlay: true,
                }),
            );
        }
    }

    /// The window's own player acts, in-frame: the window's game simulates
    /// the world and no guest's turn is on.
    fn bc_local(&self) -> bool {
        self.native_window()
            && self.remote.is_none()
            && self.lan_acting.is_none()
            && self.player.is_some()
    }

    /// The picker for the player acting now (their dimension's biomes, their
    /// search), its list filtered.
    fn bc_picker(&mut self, filter: Option<&str>) -> Option<(Actor, Tag)> {
        let (actor, feet, dimension) = self.bc_player()?;
        let entries = self.bc_entries();
        let track = shared(|s| s.tracks.get(&actor).cloned());
        let from = [f64::from(feet[0]) + 0.5, f64::from(feet[2]) + 0.5];
        let status = bc::status_line(track.as_ref(), from, dimension);
        let current = track.as_ref().map(|t| match &t.phase {
            Phase::Found { name, .. } => name.clone(),
            _ => t.label.clone(),
        });
        let dialog = bc::picker_dialog(&bc::Picker {
            entries: &entries,
            filter,
            dimension,
            status,
            current: current.as_deref(),
        });
        Some((actor, item::tag(&dialog)))
    }

    /// The picker as a dialog to a guest (single player's own window is one
    /// of its server thread's); a player who left gets nothing.
    fn bc_send_picker(&mut self, actor: Actor, dialog: Tag) {
        let Some(host) = self.lan.as_mut() else {
            return;
        };
        let to = host
            .hub
            .guests
            .iter()
            .find(|g| g.placed && uuid_ints(g.uuid) == actor)
            .map(|g| g.conn);
        if let Some(conn) = to {
            host.hub.send_to(
                conn,
                &C::ShowDialog(play::ShowDialog {
                    dialog: play::DialogHolder::Direct(dialog),
                }),
            );
        }
    }

    /// `/biomecompass pick [filter]`, and a right-click with the Biome
    /// Compass: the picker. A guest gets the dialog, at once from a command
    /// (the hub is in place), at the next tick from a right-click (the hub
    /// is out of the game during the guest's turn). The window's player
    /// in-frame, whose dialogs reach no server, gets the clickable list in
    /// chat.
    fn bc_pick(&mut self, filter: Option<&str>) {
        if self.bc_local() {
            self.bc_list(filter);
            self.bc_stop_line();
            return;
        }
        let Some((actor, dialog)) = self.bc_picker(filter) else {
            self.command_failure(Text::literal(
                "BiomeCompass's picker is for a player in a world",
            ));
            return;
        };
        if self.lan.is_some() {
            self.bc_send_picker(actor, dialog);
        } else {
            with_engine(self, |_, e| e.pickers.push((actor, dialog)));
        }
    }

    /// Under the chat picker: a link that stops the search, if one runs.
    fn bc_stop_line(&mut self) {
        let Some((actor, ..)) = self.bc_player() else {
            return;
        };
        if !shared(|s| s.tracks.contains_key(&actor)) {
            return;
        }
        let stop = Text::literal("[Stop searching]").with_style(&Style {
            click: Some(ClickEvent::RunCommand(format!("/{} stop", bc::COMMAND))),
            hover: Some(Box::new(Text::literal(
                "The compass points at the world spawn again",
            ))),
            ..Style::color(text::RED)
        });
        self.system_message(stop);
    }

    /// `/biomecompass give`: a Biome Compass, for operators and creative
    /// players (anyone else crafts one).
    fn bc_give(&mut self) {
        if self.ui.registries.permission < bc::OPERATOR && !self.creative {
            self.command_failure(Text::literal(format!(
                "BiomeCompass give needs permission level {} or creative mode; craft one: {}",
                bc::OPERATOR,
                bc::RECIPE_TEXT
            )));
            return;
        }
        if self.bc_player().is_none() {
            self.command_failure(Text::literal("BiomeCompass gives to a player in a world"));
            return;
        }
        match self.give(item::new_stack()) {
            None => self.system_message(Text::literal(format!(
                "BiomeCompass: here is a {}; right-click it to choose a biome",
                bc::ITEM_NAME
            ))),
            Some(_) => self.command_failure(Text::literal("BiomeCompass: your inventory is full")),
        }
    }

    /// `use_potion_item`'s HEAD (pinned): a right-click with the Biome
    /// Compass opens the picker, in the game that simulates the world, for
    /// the player acting (a guest's `UseItem`, or the window's player
    /// in-frame); true when it was one (the click ends there). Held down,
    /// the click repeats: the picker opens once per `USE_COOLDOWN` ticks.
    pub(super) fn biomecompass_use(&mut self, stack: &ItemStack) -> bool {
        if self.remote.is_some() || !item::is_biome_compass(stack) || !on(self) {
            return false;
        }
        let Some((actor, ..)) = self.bc_player() else {
            return false;
        };
        let open = with_engine(self, |_, e| {
            let ready = e
                .used
                .get(&actor)
                .is_none_or(|t| e.clock.wrapping_sub(*t) >= bc::USE_COOLDOWN);
            if ready {
                e.used.insert(actor, e.clock);
            }
            ready
        });
        if open {
            self.bc_pick(None);
        }
        true
    }

    /// A client of another computer's server: joined to a server that is
    /// not this window's own (single player's server thread, or the LAN
    /// world it hosts, run in this process).
    fn bc_foreign_client(&self) -> bool {
        bc::foreign_client(self.remote.is_some(), self.has_singleplayer_server())
    }

    /// `craft_recipe`'s HEAD (pinned): the window's own crafting grid
    /// holding the Biome Compass's recipe, in the game that simulates the
    /// world (in-frame) or in a window whose server is its own (single
    /// player's server thread, its LAN world), where it shows the result
    /// that server makes (`Menu::recompute`); a client of another
    /// computer's server predicts none. Its id is `repair_item`, a special
    /// recipe: the recipe book never learns one, and the grid's remainders
    /// are its items' own.
    pub(super) fn biomecompass_craft(&self) -> Option<(String, ItemStack)> {
        let foreign = self.bc_foreign_client();
        // `item::craft` reads it too (the window's first frames on a server
        // come before its first tick).
        bc::set_client_thread(foreign);
        if foreign || !on(self) {
            return None;
        }
        let n = self.craft_size;
        let grid: Vec<Option<ItemStack>> = (0..n * n).map(|i| self.craft_grid[i].clone()).collect();
        item::craft(&grid, n, self.res.items()).map(|s| ("repair_item".to_owned(), s))
    }

    /// Holding a compass in either hand.
    fn bc_holds_compass(&self) -> bool {
        let is = |s: &Option<horizon_client::items::ItemStack>| {
            s.as_ref()
                .is_some_and(|s| bc::COMPASSES.iter().any(|c| *c == &*s.item))
        };
        self.inventory.get(self.selected).is_some_and(is) || is(&self.offhand)
    }

    /// `native_hud` (a hook point): the window's box with the needle and
    /// the biome's name and distance, for its own player's search.
    pub(super) fn biomecompass_hud(&mut self, batch: &mut HudBatch, corners: &mut Corners) {
        if !on(self) {
            return;
        }
        // The search runs where the world is simulated: here in-frame, or
        // single player's server thread in this process. Another computer's
        // server keeps its searches to itself.
        if self.bc_foreign_client() {
            return;
        }
        let config = config(self);
        let Some(hud) = config.hud else {
            return;
        };
        let place = corner(hud);
        // F3's debug screen has a column in each top corner (this commit's
        // `native_corner_shown` reports only the left one taken), and other
        // mods' boxes (LookAt's) go between them: with it open, a top-left
        // or top-right box hides.
        let f3 = !self.native_corner_shown(Corner::TopLeft);
        if bc::hud_hidden_by_f3(hud, f3) || !self.native_corner_shown(place) {
            return;
        }
        let Some(p) = self.native_player() else {
            return;
        };
        let me = self.bc_window_actor();
        let Some(track) = shared(|s| s.tracks.get(&me).cloned()) else {
            return;
        };
        if config.hud_needs_compass && !self.bc_holds_compass() {
            return;
        }
        let viewer = Viewer {
            x: p.x,
            y: p.y,
            z: p.z,
            yaw: p.yaw,
            dimension: p.dimension.to_owned(),
        };
        let view = bc::hud_view(&track, &viewer);
        const DIAL: f32 = 20.0;
        let text_w = view
            .lines
            .iter()
            .map(|(t, _)| self.native_text_width(t))
            .fold(0.0, f32::max);
        let w = 2.0 + DIAL + 4.0 + text_w + 3.0;
        let h = (DIAL + 4.0).max(10.0 * view.lines.len() as f32 + 4.0);
        let (x, y) = corners.place(place, w, h);
        self.native_fill(batch, [x, y, w, h], 0x7800_0000);
        let (cx, cy) = (x + 2.0 + DIAL / 2.0, y + h / 2.0);
        self.native_fill(batch, [x + 2.0, cy - DIAL / 2.0, DIAL, DIAL], 0x5000_0000);
        let angle = match view.needle {
            Needle::Hidden => None,
            Needle::Point(a) => {
                SPIN.set(None);
                Some(a)
            }
            Needle::Spin => {
                let start = SPIN.get().unwrap_or_else(Instant::now);
                SPIN.set(Some(start));
                Some((start.elapsed().as_secs_f32() * 240.0) % 360.0)
            }
        };
        let atlas = &self.gui.atlas;
        match angle {
            Some(a) => {
                let (front, back) = bc::needle(a, [cx, cy], DIAL / 2.0 - 1.0, 2.5);
                batch.fill_quad(atlas, back, rgba(0xE0, 0xE0, 0xE0, 0xFF));
                batch.fill_quad(atlas, front, rgba(0xE8, 0x30, 0x30, 0xFF));
                batch.fill(
                    atlas,
                    cx - 1.0,
                    cy - 1.0,
                    2.0,
                    2.0,
                    rgba(0x30, 0x30, 0x30, 0xFF),
                );
            }
            None => batch.fill(
                atlas,
                cx - 1.5,
                cy - 1.5,
                3.0,
                3.0,
                rgba(0xAA, 0xAA, 0xAA, 0xFF),
            ),
        }
        let top = y + (h - 10.0 * view.lines.len() as f32) / 2.0 + 1.0;
        for (i, (line, rgb)) in view.lines.iter().enumerate() {
            let at = [x + 2.0 + DIAL + 4.0, top + 10.0 * i as f32];
            self.native_text(batch, line, at, *rgb, true);
        }
    }

    /// `native_settings` (a hook point): the settings page's buttons.
    pub(super) fn biomecompass_settings(&self) -> Vec<String> {
        config(self).labels()
    }

    /// `native_settings_click` (a hook point): change the setting, keep it
    /// in the file and put it in effect (or read the file again).
    pub(super) fn biomecompass_settings_click(&self, button: usize) {
        match bc::click(&config(self), button) {
            Some(Click::Save(c)) => {
                self.native_set_setting(bc::ID, bc::CONFIG_FILE, &c.to_file());
                publish(c);
            }
            Some(Click::Reload) => publish(read_config(self).0),
            None => {}
        }
    }

    /// `lan_actions::Game::guest_command`'s HEAD (pinned): queue a LAN or
    /// server guest's `/biomecompass` so it runs as that guest, at their
    /// own permission level, even where guests may not run commands; true
    /// when it was one.
    pub(super) fn biomecompass_guest_command(
        &mut self,
        host: &mut super::lan::LanHost,
        conn: u32,
        command: &str,
    ) -> bool {
        if !on(self) || !bc::is_command(command) {
            return false;
        }
        if let Some(permission) = host.hub.guest(conn).map(|g| g.permission) {
            host.commands.push(super::lan_actions::GuestCommand {
                conn,
                command: command.trim().trim_start_matches('/').to_owned(),
                permission,
                integrated: host.hub.config.integrated,
                quiet: false,
            });
        }
        true
    }
}

/// Keeps showing a search that found nothing for a while.
fn engine_missing(missing: &mut BTreeMap<Actor, u32>, actor: Actor) {
    missing.insert(actor, bc::MISSING_TICKS);
}
