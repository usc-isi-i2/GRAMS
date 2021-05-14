import { observable, toJS, action } from "mobx";
import { Socket } from "../library";
import { Resource } from "./Entity";

export interface GraphClassNode {
  id: string;
  uri: string;
  // for class node only, telling if the class is an approximation
  approximation: boolean;
  // readable label in form of `{label} ({qnode id})`; not obtaining from URICount.
  label: string;
}

export interface GraphDataNode {
  id: string;
  // column name
  label: string;
  columnId: number;
}

export type LiteralDataType = "entity-id" | "string";

export interface GraphLiteralNode {
  id: string;
  /**
   * URI for QNode only. Otherwise, will be an empty string
   */
  uri: string;
  // column name
  label: string;
  /**
   * DataType of the literal node
   */
  datatype: LiteralDataType;
  // whether this is a node in the context, apply for literal node only
  readonly isInContext: boolean;
}

export interface GraphNode extends GraphClassNode, GraphDataNode, GraphLiteralNode {
  // whether this is a class node; this is useful to distinguish between a literal values
  readonly isClassNode: boolean;
  // whether this is a data node
  readonly isDataNode: boolean;
  // whether this is a literal node
  readonly isLiteralNode: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  uri: string;
  approximation: boolean;
  label: string;
}

export class URICount {
  // a map from uri of nodes to the next available number
  private counter: { [uri: string]: number } = {};
  private id2num: { [id: string]: number } = {};

  constructor(nodes?: GraphNode[]) {
    for (let node of (nodes || [])) {
      if (node.isDataNode) continue;

      if (this.counter[node.uri] === undefined) {
        this.counter[node.uri] = 1;
      }
      this.id2num[node.id] = this.counter[node.uri];
      this.counter[node.uri] += 1;
    }
  }

  label = (node: GraphNode) => {
    return `${node.label} ${this.id2num[node.id]}`;
  }

  nextLabel = (uri: string, label: string) => {
    return `${label} ${this.counter[uri] || 1}`;
  }

  unlabel = (label: string) => {
    return label.substring(0, label.lastIndexOf(" "));
  }

  add = (node: GraphClassNode) => {
    if (this.counter[node.uri] === undefined) {
      this.counter[node.uri] = 1;
    }
    this.id2num[node.id] = this.counter[node.uri];
    this.counter[node.uri] += 1;
  }
}

export class Graph {
  private socket: Socket;

  public id: string;
  @observable version: number;
  @observable nodes: GraphNode[];
  @observable edges: GraphEdge[];
  // if it is stale
  @observable stale: boolean;
  @observable dataNodePositions?: { left: number; centerLeft: number }[];
  @observable nodeId2Index: { [id: string]: number } = {};
  @observable column2nodeIndex: { [columnId: number]: number } = {};
  @observable uriCount: URICount;

  constructor(socket: Socket, id: string, nodes: GraphNode[], edges: GraphEdge[]) {
    this.socket = socket;
    this.id = id;
    this.version = 0;
    this.nodes = nodes;
    this.edges = edges;
    this.stale = false;

    this.buildIndex();
    this.uriCount = new URICount(this.nodes);
  }

  onSave = () => {
    this.stale = false;
  }

  node = (id: string) => this.nodes[this.nodeId2Index[id]];
  hasNode = (id: string) => this.nodeId2Index[id] !== undefined;
  nodesByURI = (uri: string) => this.nodes.filter(node => node.uri === uri);
  nodeByColumnId = (id: number) => this.nodes[this.column2nodeIndex[id]];

  edge = (source: string, target: string) => this.edges.filter(e => e.source === source && e.target === target)[0];
  hasEdge = (source: string, target: string) => this.edges.filter(e => e.source === source && e.target === target).length > 0;
  incomingEdges = (target: string) => this.edges.filter(e => e.target === target);
  outgoingEdges = (source: string) => this.edges.filter(e => e.source === source);

  nextNodeId = () => {
    for (let i = 0; i < this.nodes.length * 100; i++) {
      let nid = `c-${i}`;
      if (this.nodeId2Index[nid] === undefined) {
        return nid;
      }
    }
    throw new Error("Cannot find new id for a node");
  }

  public toJS() {
    return {
      nodes: toJS(this.nodes),
      edges: toJS(this.edges),
      nodeId2Index: toJS(this.nodeId2Index)
    }
  }

  /** Find all paths (max 2 hops) that connect two nodes */
  findPathMax2hops = (sourceId: string, targetId: string): [GraphEdge, GraphEdge?][] => {
    let source = this.node(sourceId);
    let target = this.node(targetId);

    let matchPaths: [GraphEdge, GraphEdge?][] = [];
    let outedges = this.outgoingEdges(sourceId);
    for (let outedge of outedges) {
      if (outedge.target === targetId) {
        matchPaths.push([outedge, undefined]);
        continue;
      }

      for (let outedge2 of this.outgoingEdges(outedge.target)) {
        if (outedge2.target === targetId) {
          matchPaths.push([outedge, outedge2]);
        }
      }
    }

    return matchPaths;
  }

  /**
   * Get the class node of an entity column. Undefined if the column is not an entity node
   * @param columnId 
   * @returns 
   */
  getClassIdOfColumnId = (columnId: number): string | undefined => {
    let inedges = this.incomingEdges(this.nodeByColumnId(columnId).id);
    for (let inedge of inedges) {
      if (inedge.uri === "http://www.w3.org/2000/01/rdf-schema#label") {
        if (inedges.length > 1) {
          throw new Error("Invalid semantic model. An entity column has two incoming edges");
        }
        return inedge.source;
      }
    }
    return undefined;
  }

  getOutgoingProperties = (id: string): [GraphEdge, GraphEdge?][] => {
    let outprops: [GraphEdge, GraphEdge?][] = [];
    for (let outedge of this.outgoingEdges(id)) {
      let target = this.node(outedge.target);
      if (target.uri === 'http://wikiba.se/ontology#Statement') {
        for (let coutedge of this.outgoingEdges(outedge.target)) {
          outprops.push([outedge, coutedge]);
        }
      } else {
        outprops.push([outedge, undefined]);
      }
    }
    return outprops;
  }

  /******************************************************************
   * Below is a list of operators that modify the graph. The index is rebuilt/modify
   * inside @action function
   ******************************************************************
  */

  /**
   * Add a link between two columns
   * 
   * @deprecated
   * @param sourceColumnId
   * @param targetColumnId
   * @param edgeData
   */
  @action
  public addColumnRelationship(sourceColumnId: number, targetColumnId: number, edgeData: Omit<GraphEdge, "source" | "target">) {
    let source = this.nodeByColumnId(sourceColumnId);
    let target = this.nodeByColumnId(targetColumnId);

    let sourceIncomingEdges = this.incomingEdges(source.id);
    if (sourceIncomingEdges.length === 0) {
      throw new Error("Cannot add link from a data node to another node");
    }
    if (sourceIncomingEdges.length !== 1) {
      throw new Error("The source column connects to multiple class nodes! Don't know the exact class node to choose");
    }

    let targetIncomingEdges = this.incomingEdges(target.id);
    if (targetIncomingEdges.length > 1) {
      throw new Error("The target column connects to multiple class nodes! Don't know the exact class node to choose");
    }

    let realSource = sourceIncomingEdges[0].source;
    let realTarget = targetIncomingEdges.length === 0 ? target.id : targetIncomingEdges[0].source;

    this.addEdge({
      ...edgeData,
      source: realSource,
      target: realTarget
    });
  }

  /**
   * Upsert the type of the column: replace the type if exist otherwise, create the type including the
   * new class node.
   *
   * @param columnId
   * @param source
   */
  @action
  public upsertColumnType(columnId: number, source: Omit<GraphClassNode, "id">) {
    let target = this.nodeByColumnId(columnId);
    let targetIncomingEdges = this.incomingEdges(target.id);

    if (targetIncomingEdges.length > 1) {
      throw new Error("The column connects to multiple class nodes! Don't know the exact class node to choose");
    }

    if (targetIncomingEdges.length === 0) {
      let sourceId = this.nextNodeId();
      this.addClassNode({
        ...source,
        id: sourceId,
      });

      this.addEdge({
        source: sourceId,
        target: target.id,
        uri: "http://www.w3.org/2000/01/rdf-schema#label",
        label: "rdfs:label",
        approximation: false,
      });
    } else {
      let edge = targetIncomingEdges[0];
      this.updateNode(edge.source, source);
      if (edge.uri !== "http://www.w3.org/2000/01/rdf-schema#label") {
        // need to update the edge as well
        this.updateEdge(edge.source, edge.target, {
          uri: "http://www.w3.org/2000/01/rdf-schema#label",
          label: "rdfs:label",
          approximation: edge.approximation
        });
      }
    }
  }

  /**
   * Upsert the relationship between two nodes: replace the type if exist otherwise create id.
   * 
   * This is a special function as it tight the system to Wikidata with special node of
   * wikibase:Statement & property/qualifier. Assuming that the source node and target node
   * always exist.
   * 
   * @param sourceId 
   * @param targetId 
   * @param pred1
   * @param pred2
   */
  @action
  public upsertRelationship(sourceId: string, targetId: string, pred1: Resource, pred2: Resource) {
    let source = this.node(sourceId);
    let target = this.node(targetId);

    let matchPaths = this.findPathMax2hops(sourceId, targetId);

    if (matchPaths.length === 0) {
      // no new node, so we need to create it
      if (pred1.uri === pred2.uri) {
        // we just need to create one link
        this.addEdge({
          source: sourceId,
          target: targetId,
          uri: pred1.uri,
          label: pred1.label,
          approximation: false
        });
      } else {
        let tempid = this.nextNodeId();
        this.addClassNode({
          id: tempid,
          uri: 'http://wikiba.se/ontology#Statement',
          label: 'wikibase:Statement',
          approximation: false,
        });
        this.addEdge({
          source: sourceId,
          target: tempid,
          uri: pred1.uri,
          label: pred1.label,
          approximation: false
        });
        this.addEdge({
          source: tempid,
          target: targetId,
          uri: pred2.uri,
          label: pred2.label,
          approximation: false
        });
      }
      return;
    }

    if (matchPaths.length > 1) {
      throw new Error("There are more one path between two nodes. Don't know which path to update it");
    }

    let [edge1, edge2] = matchPaths[0];
    this.updateEdge(edge1.source, edge1.target, {
      uri: pred1.uri,
      label: pred1.label,
      approximation: false
    });
    if (edge2 !== undefined) {
      // non direct property, we need to update it as well
      this.updateEdge(edge2.source, edge2.target, {
        uri: pred2.uri,
        label: pred2.label,
        approximation: false
      });
    }
  }

  /**
   * Add a class node to the model.
   */
  @action
  public addClassNode(node: GraphClassNode) {
    if (this.nodeId2Index[node.id] !== undefined) {
      throw new Error("Duplicated id");
    }
    this.nodeId2Index[node.id] = this.nodes.length;
    this.nodes.push({
      ...node,
      isClassNode: true,
      isDataNode: false,
      isLiteralNode: false,
      isInContext: false,
      datatype: "string",
      columnId: -1,
    });
    this.uriCount.add(node);
    this.version += 1;
    this.stale = true;
  }

  /**
   * Add a literal node to the model
   */
  @action
  public addLiteralNode(node: GraphLiteralNode) {
    if (this.nodeId2Index[node.id] !== undefined) {
      throw new Error("Duplicated id");
    }
    this.nodeId2Index[node.id] = this.nodes.length;
    this.nodes.push({
      ...node,
      approximation: false,
      isClassNode: false,
      isDataNode: false,
      isLiteralNode: true,
      columnId: -1,
    });
    this.version += 1;
    this.stale = true;
  }

  @action
  public removeNode(nodeId: string) {
    this._removeNode(nodeId);
    this.nodes = this.nodes.filter(n => n !== undefined);
    this.buildIndex();
    this.version += 1;
    this.stale = true;
    this.uriCount = new URICount(this.nodes);
  }

  @action
  public updateNode(nodeId: string, props: Partial<GraphNode>) {
    let nodeIndex = this.nodeId2Index[nodeId];
    this.nodes[nodeIndex] = { ...this.nodes[nodeIndex], ...props };

    this.version += 1;
    this.stale = true;
    if (props.uri !== undefined) {
      this.uriCount = new URICount(this.nodes);
    }
  }

  @action
  public addEdge(edge: GraphEdge) {
    if (this.edges.filter(e => e.source === edge.source && e.target === edge.target).length > 0) {
      throw new Error("Cannot have more than one edge between two classes");
    }

    this.edges.push(edge);
    this.version += 1;
    this.stale = true;
  }

  @action
  public removeEdge(sourceId: string, targetId: string) {
    let size = this.nodes.length;
    this._removeEdge(sourceId, targetId);
    this.nodes = this.nodes.filter(n => n !== undefined);

    if (this.nodes.length !== size) {
      this.buildIndex();
      this.uriCount = new URICount(this.nodes);
    }
    this.version += 1;
    this.stale = true;
  }

  @action
  public updateEdge(source: string, target: string, props: Partial<GraphEdge>) {
    for (let i = 0; i < this.edges.length; i++) {
      let edge = this.edges[i];
      if (edge.source === source && edge.target === target) {
        this.edges[i] = { ...this.edges[i], ...props };
      }
    }
    this.version = (this.version || 0) + 1;
    this.stale = true;
  }

  /**
   * Cascading remove nodes in the graph. To avoid rebuilding the index
   * everytime we delete a node, we replace the deleted node by undefined.
   * A post process step is needed to filter out all undefined items in the list
   */
  private _removeNode = (nodeId: string) => {
    if (this.nodeId2Index[nodeId] === undefined || this.nodes[this.nodeId2Index[nodeId]] === undefined) {
      return;
    }

    let nodeIndex = this.nodeId2Index[nodeId];
    if (this.nodes[nodeIndex].isDataNode || this.nodes[nodeIndex].isInContext) {
      // don't remove data nodes && context node;
      return;
    }

    // remove node by mark it undefined
    (this.nodes[nodeIndex] as any) = undefined;

    // we need to remove other edges connected to this node
    let edges = this.edges.filter(edge => edge.source === nodeId || edge.target === nodeId);
    for (let edge of edges) {
      this._removeEdge(edge.source, edge.target);
    }
  }

  /**
   * Cascading remove edges in the graph
   */
  private _removeEdge = (sourceId: string, targetId: string) => {
    let edgeIndex = undefined;
    let sourceDegree = 0;
    let targetDegree = 0;

    for (let i = 0; i < this.edges.length; i++) {
      let edge = this.edges[i];
      if (edge.source === sourceId && edge.target === targetId) {
        edgeIndex = i;
      }
      if (edge.source === sourceId || edge.target === sourceId) {
        sourceDegree += 1;
      }
      if (edge.source === targetId || edge.target === targetId) {
        targetDegree += 1;
      }
    }

    if (edgeIndex === undefined) {
      return;
    }

    // remove edge
    this.edges.splice(edgeIndex, 1);

    // if a node only has one connection, so we delete it because now it is lonely, except if it is
    // a data node
    if (sourceDegree === 1) {
      this._removeNode(sourceId);
    }
    if (targetDegree === 1) {
      this._removeNode(targetId);
    }
  }

  private buildIndex = () => {
    this.nodeId2Index = {};
    this.column2nodeIndex = {};

    for (let i = 0; i < this.nodes.length; i++) {
      let n = this.nodes[i];
      this.nodeId2Index[n.id] = i;
      if (n.columnId !== null) {
        this.column2nodeIndex[n.columnId] = i;
      }
    }
  }
}