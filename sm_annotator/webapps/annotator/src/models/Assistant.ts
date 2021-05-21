import { flow, observable } from "mobx";
import { CancellablePromise } from "mobx/lib/api/flow";
import { Socket } from "../library";
import { Resource } from "./Entity";

export interface TypeHierarchy {
  uri: string;
  label: string;
  depth: number;
  duplicated: boolean;
  freq: number;
  percentage: number;
}

export interface Relationship {
  endpoint: Resource | number;
  predicates: [Resource, Resource];
  freq: number;
}

export class ColumnAssistant {
  @observable stats?: { [metric: string]: string };
  @observable flattenTypeHierarchy?: TypeHierarchy[];
  @observable type2children?: { [uri: string]: string[] };
  @observable relationships?: { incoming: Relationship[], outgoing: Relationship[] };

  constructor(
    stats?: { [metric: string]: string },
    flattenTypeHierarchy?: TypeHierarchy[],
    type2children?: { [uri: string]: string[] },
    relationships?: { incoming: Relationship[], outgoing: Relationship[] }
  ) {
    this.stats = stats;
    this.flattenTypeHierarchy = flattenTypeHierarchy;
    this.type2children = type2children;
    this.relationships = relationships;
  }
}

export class Assistant {
  private socket: Socket;

  public id: string;
  @observable columns: { [id: string]: ColumnAssistant };
  @observable loading: boolean;

  constructor(socket: Socket, id: string) {
    this.socket = socket;
    this.id = id;
    this.columns = {};
    this.loading = false;
  }

  public requestColumnSuggestions = flow(function* (this: Assistant, columnIndex: number) {
    this.loading = true;
    let resp = yield this.socket.request("/assistant/column", {
      columnIndex: columnIndex
    });
    this.columns[columnIndex] = new ColumnAssistant(resp.response.stats, resp.response.flattenTypeHierarchy, resp.response.type2children, resp.response.relationships);
    this.loading = false;
  })
}