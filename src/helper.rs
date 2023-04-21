use hashbrown::HashMap;
use std::borrow::Borrow;
use std::cmp::Eq;
use std::hash::Hash;
use std::ops::Index;

pub struct ByValue;
pub struct ByReference;

pub trait ReturnKind<'a, T: Sized + 'a> {
    type Type: Sized;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByValue {
    type Type = T;
}

impl<'a, T: Sized + 'a> ReturnKind<'a, T> for ByReference {
    type Type = &'a T;
}

/// A map that allow to access a node by both position and key
pub struct PositionMap<K: Eq + Hash, V> {
    data: Vec<V>,
    index: HashMap<K, usize>,
}

impl<K: Eq + Hash, V> PositionMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) {
        let index = self.data.len();
        self.data.push(value);
        self.index.insert(key, index);
    }

    pub fn get_pos<Q: ?Sized>(&self, k: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).cloned()
    }

    pub fn get<Q: ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).map(|&index| &self.data[index])
    }

    pub fn get_mut<Q: ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.index.get(k).map(|&index| &mut self.data[index])
    }
}

impl<K: Eq + Hash, V> Index<usize> for PositionMap<K, V> {
    type Output = V;

    fn index(&self, index: usize) -> &V {
        &self.data[index]
    }
}
