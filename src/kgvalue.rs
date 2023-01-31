pub enum EntityType {
    Item,
    Property,
}

pub struct EntityId {
    id: String,
    entity_type: EntityType,
    numeric_id: String,
}

pub struct Time {
    // See more: https://doc.wikimedia.org/Wikibase/master/php/md_docs_topics_json.html
    time: String,
    timezone: u64,
    before: u64,
    after: u64,
    precision: u64,
    calendarmodel: String,
}

pub struct Quantity {
    // The nominal value of the quantity, as an arbitrary precision decimal string. The string always starts with a character indicating the sign of the value, either “+” or “-”.
    amount: String,
    // Optionally, the upper bound of the quantity's uncertainty interval, using the same notation as the amount field. If not given or null, the uncertainty (or precision) of the quantity is not known. If the upperBound field is given, the lowerBound field must also be given.
    upper_bound: String,
    // Optionally, the lower bound of the quantity's uncertainty interval, using the same notation as the amount field. If not given or null, the uncertainty (or precision) of the quantity is not known. If the lowerBound field is given, the upperBound field must also be given.
    lower_bound: String,
    // The URI of a unit (or “1” to indicate a unit-less quantity). This would typically refer to a data item on wikidata.org, e.g. http://www.wikidata.org/entity/Q712226 for “square kilometer”.
    unit: String,
}

pub struct GlobeCoordinate {
    latitude: f64,
    longitude: f64,
    precision: f64,
    altitude: Option<f64>,
    // The URI of a reference globe. This would typically refer to a data item on wikidata.org. This is usually just an indication of the celestial body (e.g. Q2 = earth), but could be more specific, like WGS 84 or ED50.
    globe: String,
}

pub struct MonolingualText {
    text: String,
    language: String,
}

pub enum KGValue {
    String(String),
    EntityId(EntityId),
    Time(Time),
    Quantity(Quantity),
    MonolingualText(MonolingualText),
    GlobeCoordinate(GlobeCoordinate),
}