import { Table } from "./Table";
import { Graph } from "./Graph";
import { Socket, Store } from "../library";
import { action, toJS, computed } from "mobx";
import { Assistant } from "./Assistant";
import { EntityStore } from "./Entity";

export interface AnnotationLog {
  note: string;
  isCurated: boolean;
  stale: boolean;
}

export interface StoreProps {
  table: Table;
  log: AnnotationLog,
  graphs: Graph[];
  currentGraphIndex: number;
  assistant: Assistant;
  entities: EntityStore;
  wdOntology: {
    url: string;
    username: string;
    password: string;
  };
}

export class AppStore extends Store<StoreProps> {
  constructor(socket: Socket, defaultProps: StoreProps) {
    super(socket, defaultProps, undefined, AppStore.deserialize);
  }

  findClassByName = (query: string): Promise<{ uri: string, label: string, description: string }[]> => {
    return this.socket.request("/ontology/class", { query }).then((resp) => resp.response as { uri: string, label: string, description: string }[]);
  }

  findPredicateByName = (query: string): Promise<{ uri: string, label: string, description: string }[]> => {
    return this.socket.request("/ontology/predicate", { query }).then((resp) => resp.response as { uri: string, label: string, description: string }[]);
  }

  /**
   * Whether the content of the store is out of sync with the server
   */
  get stale(): boolean {
    return this.props.log.stale || this.props.graphs.some((g) => g.stale);
  }

  /**
   * Get the current annotating graph
   */
  get currentGraph(): Graph {
    return this.props.graphs[this.props.currentGraphIndex];
  }

  @computed get relevantProps(): string[] {
    let props: Set<string> = new Set();
    for (let col of Object.values(this.props.assistant.columns)) {
      if (col.relationships !== undefined) {
        for (let rel of col.relationships.incoming) {
          props.add(rel.predicates[0].uri);
          props.add(rel.predicates[1].uri)
        }
        for (let rel of col.relationships.outgoing) {
          props.add(rel.predicates[0].uri);
          props.add(rel.predicates[1].uri)
        }
      }
    }

    for (let g of this.props.graphs) {
      for (let edge of g.edges) {
        props.add(edge.uri);
      }
    }
    return Array.from(props);
  }

  /**
   * Set a particular graph to view
   *
   * @param currentGraphIndex index of the graph we want to view
   */
  @action
  public setCurrentGraph(currentGraphIndex: number) {
    this.props.currentGraphIndex = currentGraphIndex;
  }

  /**
   * Create a new graph and show it (moving current index)
   */
  @action
  public createNewGraphFromCurrentGraph() {
    let g = this.currentGraph;
    let newG = new Graph(this.socket, `G-${this.props.table.id}-${this.props.graphs.length}`, toJS(g.nodes), toJS(g.edges));
    this.props.currentGraphIndex = this.props.graphs.length;
    this.props.graphs.push(newG);
  }

  /**
   * Remove current graph
   */
  @action
  public removeCurrentGraph() {
    this.props.graphs.splice(this.props.currentGraphIndex, 1);
    this.props.currentGraphIndex = 0;
    // need someway to tell the current graph has change
    this.props.log.stale = true;
  }

  /**
   * Save store to the server
   */
  @action
  public save() {
    return this.socket.request("/save", {
      isCurated: toJS(this.props.log.isCurated),
      note: toJS(this.props.log.note),
      graphs: this.props.graphs.map(g => ({
        nodes: toJS(g.nodes),
        edges: toJS(g.edges),
      }))
    }).then(() => {
      this.props.graphs.forEach(g => {
        g.stale = false;
      });
      this.props.log.stale = false;
    });
  }

  @action
  public updateNote(note: string) {
    this.props.log.note = note;
    this.props.log.stale = true;
  }

  @action
  public updateIsCurated(isCurated: boolean) {
    this.props.log.isCurated = isCurated;
    this.props.log.stale = true;
  }

  // deserialize the data from the server
  public static deserialize(socket: Socket, prop: string, data: any) {
    if (prop === "graphs") {
      return data.map((g: any, idx: number) => new Graph(socket, `G-${g.tableID}-${idx}`, g.nodes, g.edges));
    }
    if (prop === "table") {
      return new Table(socket, data);
    }
    if (prop === "assistant") {
      return new Assistant(socket, data.id);
    }
    if (prop === "entities") {
      return new EntityStore(socket, data);
    }
    return data;
  }
}