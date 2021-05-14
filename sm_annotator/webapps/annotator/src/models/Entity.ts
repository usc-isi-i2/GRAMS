import { flow, observable } from "mobx";
import { Socket } from "../library";
import { CancellablePromise } from "mobx/lib/api/flow";

/**
 * Resource could be predicate, class, or entity
 */
export interface Resource {
  uri: string;
  label: string;
}

interface TypeHierarchy extends Resource {
  depth: number;
}

interface PropertyValues {
  uri: string;
  label: string;
  values: {
    value: string | Resource;
    qualifiers: {
      [uri: string]: {
        uri: string;
        label: string;
        values: (string | Resource)[]
      }
    }
  }[];
}

export interface ExcerptEntity {
  uri: string;
  label: string;
  types: TypeHierarchy[];
}

export interface Entity {
  uri: string;
  label: string;
  description: string;
  types: TypeHierarchy[];
  props: {
    [uri: string]: PropertyValues
  }
}

export class EntityStore {
  private socket: Socket;

  // use an incremental version so that we can memoize expensive query
  @observable version: number;
  @observable loading: boolean;
  @observable entities: { [uri: string]: Entity | null };

  constructor(socket: Socket, ents: { [uri: string]: Entity | null }) {
    this.socket = socket;
    this.entities = ents;
    this.loading = false;
    this.version = 0;
  }

  public getMissingEntities = (uris: string[]) => {
    return uris.filter(uri => this.entities[uri] === undefined);
  }

  public fetchData: (
    uris: string[]
  ) => CancellablePromise<void> = flow(function* (
    this: EntityStore,
    uris: string[]
  ) {
    this.loading = true;
    let resp = yield this.socket.request("/entities", { uris });
    for (let uri of uris) {
      this.entities[uri] = resp.response[uri];
    }
    this.version += 1;
    this.loading = false;
  });

  /**
   * Get an entity. null if it doesn't exist, undefined if it is not fetched to the local yet
   */
  public getEntity = (uri: string): Entity | null | undefined => {
    return this.entities[uri];
  }
}