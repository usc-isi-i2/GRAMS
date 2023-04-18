use crate::{context::AlgoContext, error::GramsError};
use anyhow::Result;
use kgdata::models::Value;
use regex::Regex;

use super::{
    parsed_text_repr::{ParsedDatetimeRepr, ParsedTextRepr},
    SingleTypeMatcher,
};

pub struct TimeTest {
    valid_calendars: [String; 2],
    kg_timestring_parser: Regex,
}

impl TimeTest {
    pub fn default() -> Self {
        TimeTest {
            valid_calendars: [
                "http://www.wikidata.org/entity/Q1985786".to_owned(),
                "http://www.wikidata.org/entity/Q1985727".to_owned(),
            ],
            kg_timestring_parser: Regex::new(r"([-+]\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z")
                .unwrap(),
        }
    }
}

impl SingleTypeMatcher for TimeTest {
    fn get_name(&self) -> &'static str {
        "time_test"
    }

    fn compare(
        &self,
        query: &ParsedTextRepr,
        key: &Value,
        _context: &AlgoContext,
    ) -> Result<(bool, f64), GramsError> {
        let celldt = match &query.datetime {
            Some(dt) => dt,
            None => return Ok((false, 0.0)),
        };

        let kgtime = &key.as_time().unwrap();
        let timestr = &kgtime.time;

        // there are two calendar:
        // gregorian calendar, and julian calendar (just 13 days behind gregorian)
        // https://www.timeanddate.com/calendar/julian-gregorian-switch.html#:~:text=13%20Days%20Behind%20Today,days%20behind%20the%20Gregorian%20calendar.
        // TODO: we need to consider a range for julian calendar
        if !self.valid_calendars.contains(&kgtime.calendarmodel) {
            return Err(GramsError::IntegrityError(format!(
                "Invalid calendar: {}",
                kgtime.calendarmodel
            )));
        }

        // TODO: handle timezone, before/after and precision
        // pass

        // parse timestring
        let m = self.kg_timestring_parser.captures(timestr).ok_or_else(|| {
            GramsError::IntegrityError(format!("Invalid timestring: {}", timestr))
        })?;
        let mut kgdt = ParsedDatetimeRepr {
            year: Some(m.get(1).unwrap().as_str().parse::<i64>().unwrap()),
            month: Some(m.get(2).unwrap().as_str().parse::<i64>().unwrap()),
            day: Some(m.get(3).unwrap().as_str().parse::<i64>().unwrap()),
            hour: Some(m.get(4).unwrap().as_str().parse::<i64>().unwrap()),
            minute: Some(m.get(5).unwrap().as_str().parse::<i64>().unwrap()),
            second: Some(m.get(6).unwrap().as_str().parse::<i64>().unwrap()),
        };

        if timestr.chars().next().unwrap() == '-' {
            kgdt.year = Some(-kgdt.year.unwrap());
        }
        if Some(0) == kgdt.year {
            kgdt.year = None;
        }
        if Some(0) == kgdt.month {
            kgdt.month = None;
        }
        if Some(0) == kgdt.day {
            kgdt.day = None;
        }
        if Some(0) == kgdt.hour {
            kgdt.hour = None;
        }
        if Some(0) == kgdt.minute {
            kgdt.minute = None;
        }
        if Some(0) == kgdt.second {
            kgdt.second = None;
        }

        if kgdt == *celldt {
            return Ok((true, 1.0));
        }
        if celldt.has_only_year() && kgdt.year == celldt.year {
            return Ok((true, 0.8));
        }
        if celldt.first_day_of_year() && kgdt.year == celldt.year {
            return Ok((true, 0.75));
        }
        return Ok((false, 0.0));
    }
}
