// BiomeCompass's item, recipe and commands on the game's own types (Horizon
// 142388e1c136e622652ec7505af15d8ad195e041, see horizon-mixin.toml). Injected
// into the game's library as `horizon_client::biomecompass_item`, beside
// `horizon_client::biomecompass` (runtime.rs), so the hooks in the library (a
// guest's crafting table, the command tree a player receives) and those in the
// game (the window's crafting, a right-click, the search) share it.
//
// A Biome Compass is a vanilla `minecraft:compass` with BiomeCompass's mark in
// `minecraft:custom_data`, a name and lore; its needle is vanilla's
// `minecraft:lodestone_tracker`, which every client already draws. Nothing is
// added to the game's registries, so vanilla clients, servers and saves read it.
use crate::biomecompass::{self as bc, Nbt, Pointing};
use crate::items::{ItemStack, Items};
use horizon_world::nbt::{JavaString, Tag};

/// BiomeCompass's NBT as the game's.
pub fn tag(n: &Nbt) -> Tag {
    match n {
        Nbt::Byte(v) => Tag::Byte(*v),
        Nbt::Int(v) => Tag::Int(*v),
        Nbt::String(s) => Tag::String(JavaString::from(s.as_str())),
        Nbt::IntArray(v) => Tag::IntArray(v.clone()),
        Nbt::List(v) => Tag::List(v.iter().map(tag).collect()),
        Nbt::Compound(entries) => Tag::compound(entries.iter().map(|(k, v)| (k.as_str(), tag(v)))),
    }
}

/// The game's NBT as BiomeCompass's (None for what it never writes: longs,
/// floats, byte arrays...).
pub fn nbt(t: &Tag) -> Option<Nbt> {
    Some(match t {
        Tag::Byte(v) => Nbt::Byte(*v),
        Tag::Int(v) => Nbt::Int(*v),
        Tag::String(s) => Nbt::String(s.to_string_lossy()),
        Tag::IntArray(v) => Nbt::IntArray(v.clone()),
        Tag::List(v) => Nbt::List(v.iter().map(nbt).collect::<Option<Vec<_>>>()?),
        Tag::Compound(map) => Nbt::Compound(
            map.iter()
                .filter_map(|(k, v)| Some((k.to_string_lossy(), nbt(v)?)))
                .collect(),
        ),
        _ => return None,
    })
}

/// A new Biome Compass, pointing nowhere yet (at the world spawn).
pub fn new_stack() -> ItemStack {
    let mut s = ItemStack::new(bc::ITEM, 1);
    let d = s.data_mut();
    d.custom_name = Some(tag(&bc::item_name()));
    d.set_other(bc::ITEM_NAME_COMPONENT, Some(tag(&bc::item_name())));
    d.set_other(bc::CUSTOM_DATA, Some(tag(&bc::custom_data())));
    d.set_other(bc::LORE, Some(tag(&bc::lore(&Pointing::Idle))));
    s
}

/// Whether `s` is a Biome Compass.
pub fn is_biome_compass(s: &ItemStack) -> bool {
    let Some(d) = s.data() else {
        return false;
    };
    bc::is_biome_compass(&s.item, &|id| d.other(id).and_then(nbt))
}

/// Brings a Biome Compass up to date with what it points at: its lodestone
/// tracker, its lore, and the mark (put back on a copy that lost it). True
/// when it changed.
pub fn point(s: &mut ItemStack, pointing: &Pointing) -> bool {
    let tracker = bc::tracker(pointing).map(|n| tag(&n));
    let lore = tag(&bc::lore(pointing));
    let d = s.data_mut();
    let mut changed = false;
    if d.other(bc::TRACKER) != tracker.as_ref() {
        d.set_other(bc::TRACKER, tracker);
        changed = true;
    }
    if d.other(bc::LORE) != Some(&lore) {
        d.set_other(bc::LORE, Some(lore));
        changed = true;
    }
    let marked = d
        .other(bc::CUSTOM_DATA)
        .and_then(|c| c.get(bc::MARK))
        .is_some();
    if !marked {
        let mut data = match d.other(bc::CUSTOM_DATA) {
            Some(t @ Tag::Compound(_)) => t.clone(),
            _ => Tag::Compound(Default::default()),
        };
        if let Tag::Compound(map) = &mut data {
            map.insert(JavaString::from(bc::MARK), Tag::Byte(1));
        }
        d.set_other(bc::CUSTOM_DATA, Some(data));
        changed = true;
    }
    changed
}

/// The Biome Compass when `grid` (`size` by `size`, row-major) holds its
/// recipe; saplings are `#minecraft:saplings` of the data pack.
pub fn craft(grid: &[Option<ItemStack>], size: usize, items: &Items) -> Option<ItemStack> {
    if !bc::enabled() || bc::client_thread() || size != 3 || grid.len() != 9 {
        return None;
    }
    let ids: Vec<Option<&str>> = grid.iter().map(|c| c.as_ref().map(|s| &*s.item)).collect();
    let saplings = items.item_tag_order.get(bc::SAPLINGS);
    let is_sapling = |id: &str| match saplings {
        Some(list) if !list.is_empty() => list.iter().any(|s| s == id),
        _ => bc::looks_like_sapling(id),
    };
    bc::recipe_matches(&ids, size, &is_sapling).then(new_stack)
}

/// BiomeCompass's commands in a root of the commands report (`lan::commands`,
/// the form server mods' commands take): `/biomecompass` and `/bcompass`, so
/// a player's chat box knows them, and a vanilla client runs the picker's
/// buttons (commands) without asking first.
pub fn add_commands(tree: &mut serde_json::Value) {
    use serde_json::{Value, json};
    if !bc::enabled() {
        return;
    }
    if !tree.is_object() {
        *tree = json!({"type": "root", "children": {}});
    }
    if !tree["children"].is_object() {
        tree["children"] = json!({});
    }
    let literal = |executable: bool, children: Value| json!({"type": "literal", "executable": executable, "children": children});
    let text = |kind: &str| {
        json!({
            "type": "argument",
            "parser": "brigadier:string",
            "properties": {"type": kind},
            "executable": true,
        })
    };
    let node = literal(
        true,
        json!({
            "pick": literal(true, json!({ "filter": text("phrase") })),
            "list": literal(true, json!({ "filter": text("greedy") })),
            "find": literal(false, json!({ "biome": text("greedy") })),
            "status": literal(true, json!({})),
            "stop": literal(true, json!({})),
            "give": literal(true, json!({})),
            "tp": literal(true, json!({})),
            "reload": literal(true, json!({})),
            "help": literal(true, json!({})),
        }),
    );
    if let Some(children) = tree["children"].as_object_mut() {
        for name in [bc::COMMAND, bc::ALIAS] {
            children
                .entry(name.to_owned())
                .or_insert_with(|| node.clone());
        }
    }
}
