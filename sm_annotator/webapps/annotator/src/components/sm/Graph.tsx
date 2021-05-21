import G6 from "@antv/g6";
import { THEME } from "../../env";
import { GraphEdge, GraphNode, URICount } from "../../models";


interface G6GraphProps {
  // init height of the canvas
  initHeight: number,
  // shift the node in the graph by `leftOffset` units
  leftOffset: number,
  onNodeClick?: (nodeId: string) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
}

export const GraphNodeColors = {
  DataNode: {
    fill: "#ffd666",
    stroke: "#874d00",
  },
  ClassNode: {
    fill: '#b7eb8f',
    stroke: '#135200',
    // fill: "#d9d9d9",
    // stroke: "#434343",
  },
  // literal nodes that are in the context
  LiteralContextNode: {
    fill: "#C6E5FF",
    stroke: "#5B8FF9",
  },
  // literal nodes that are not in the context
  LiteralNonContextNode: {
    fill: "#d3adf7",
    stroke: "#c41d7f",
  }
}

export interface G6GraphNode extends GraphNode {
  x?: number;
  y?: number;
  type?: string;
  labelCfg?: object;
  style?: any;
  size?: number | number[];
}

interface G6GraphData {
  nodes: G6GraphNode[],
  edges: GraphEdge[],
  subNodes?: G6GraphNode[],
  subEdges?: GraphEdge[],
}

export class G6Graph {
  private version: number;
  private container: HTMLDivElement;
  private graph: any;
  private data: G6GraphData;
  private props: G6GraphProps;

  constructor(
    container: HTMLDivElement,
    props: G6GraphProps
  ) {
    this.version = 0;
    this.props = props;
    this.container = container;
    this.data = {
      nodes: [],
      edges: [],
    };

    this.graph = new G6.Graph({
      container: container,
      width: container.scrollWidth,
      height: props.initHeight,
      linkCenter: true,
      layout: {
        type: 'dagre',
        rankdir: 'TB',
        nodesep: 70,
        ranksep: 30
      },
      defaultEdge: {
        type: "quadratic",
        style: {
          stroke: THEME === "dark" ? "white" : "black",
          endArrow: {
            fill: THEME === "dark" ? "white" : "black",
            path: G6.Arrow.triangle(6, 8, 15),
            d: 15
          },
        },
        labelCfg: {
          style: {
            fill: 'black',
            background: {
              fill: "#ffffff",
              stroke: "#9EC9FF",
              padding: [2, 2, 2, 2],
              radius: 2,
            },
          },
        },
      },
      defaultNode: {
        // labelCfg: {
        //   style: {
        //     fill: 'black',
        //     background: {
        //       fill: "#bfbfbf",
        //       stroke: "#5b8ef9",
        //       padding: [2, 2, 2, 2],
        //       radius: 2,
        //     },
        //   },
        // },
      },
      modes: {
        default: ["drag-canvas", "drag-node"],
        edit: ["click-select"],
      },
      nodeStateStyles: {
        hover: {
          fill: "lightsteelblue",
        },
      },
    });

    if (props.onNodeClick !== undefined) {
      this.graph.on('node:click', (event: any) => {
        props.onNodeClick!(event.item._cfg.model.id);
      });
    }

    if (props.onEdgeClick !== undefined) {
      this.graph.on('edge:click', (event: any) => {
        props.onEdgeClick!(event.item._cfg.model);
      });
    }
  }

  static transformData = (nodes: GraphNode[], edges: GraphEdge[], uriCount: URICount): G6GraphData => {
    // get new nodes
    let newNodes = nodes.map(u => {
      let n: G6GraphNode = { ...u };
      if (u.isDataNode) {
        n.labelCfg = {
          style: {
            fill: 'black',
            background: {
              ...GraphNodeColors.DataNode,
              padding: [4, 4, 4, 4],
              radius: 3,
            }
          },
        };
        n.type = 'rect';
        n.size = [1, 1];
        n.style = {
          radius: 3,
          ...GraphNodeColors.DataNode,
        };
        if (n.label === "") {
          n.label = " ";
        }
      } else if (n.isClassNode) {
        // n.type = 'ellipse';
        // n.size = [n.label.length * 6, 25];
        n.labelCfg = {
          style: {
            fill: 'black',
            background: {
              ...GraphNodeColors.ClassNode,
              padding: [4, 4, 4, 4],
              radius: 3,
            }
          },
        };
        n.style = {
          radius: 3,
          ...GraphNodeColors.ClassNode,
        };
        n.size = [30, 30];
        n.label = uriCount.label(n);
      } else if (n.isInContext) {
        n.labelCfg = {
          style: {
            fill: 'black',
            background: {
              ...GraphNodeColors.LiteralContextNode,
              padding: [4, 4, 4, 4],
              radius: 3,
            }
          },
        };
        n.style = {
          radius: 3,
          ...GraphNodeColors.LiteralContextNode,
        };
        n.size = [30, 30];
      } else {
        n.labelCfg = {
          style: {
            fill: 'black',
            background: {
              ...GraphNodeColors.LiteralNonContextNode,
              padding: [4, 4, 4, 4],
              radius: 3,
            }
          },
        };
        n.style = {
          radius: 3,
          ...GraphNodeColors.LiteralNonContextNode,
        };
        n.size = [30, 30];
      }
      return n;
    });

    // get new edges
    let newEdges = edges.map(e => ({ ...e, label: wrapTextPreserveWord(e.label, 120, 14) }));

    return { nodes: newNodes, edges: newEdges };
  }

  setDataAndRender = (data: G6GraphData, onFinish?: () => void) => {
    if (this.version > 0) {
      this.graph.clear();
    }
    this.data = data;
    this.graph.data({ nodes: data.nodes, edges: data.edges });
    this.graph.render();
    this.version += 1;

    if (onFinish !== undefined) {
      onFinish();
    }
  }

  refreshPositions = () => {
    this.graph.refreshPositions();
  }

  fitToCanvas = (center?: boolean) => {
    // follow the code in fitView & fitCenter
    let group = this.graph.get("group");
    group.resetMatrix();
    let bbox = group.getCanvasBBox();
    if (bbox.width === 0 || bbox.height === 0) return;
    // let graphWidth = this.graph.getWidth();
    let graphWidth = this.container.clientWidth;

    if (center === true) {
      this.graph.moveTo(graphWidth / 2 - bbox.width / 2, 10);
    } else {
      this.graph.moveTo(this.props.leftOffset, 10);
    }
    this.graph.changeSize(graphWidth, bbox.height + 20);
  }

  /** Render the subgraph with the force layout as we fixed the data node in place
   *
   * @deprecated
  */
  subgraphLayoutForce = (version: number, data: G6GraphData, onFinish: () => void) => {
    let onLayoutEnd = false;
    const sublayout = new G6.Layout.force({
      center: [this.container.scrollWidth / 2, 0],
      linkDistance: 100,
      preventOverlap: true,
      nodeSize: 10,
      // leave an empty function here, just to mark that the correct function is tick
      // not onTick
      tick: () => { },
      onLayoutEnd: () => {
        onLayoutEnd = true;
        if (this.version === version) {
          // run for the last time
          onFinish();
        }
      },
    });
    sublayout.init({ nodes: data.subNodes, edges: data.subEdges });
    sublayout.execute();

    // update position every 50ms, to make it looks smooth,
    // unless the layout is finished rendering
    let fn = () => {
      // console.log('update position');
      if (this.version > version) {
        // console.log('detect newer code - stop previous code');
        return;
      }
      onFinish();
      if (!onLayoutEnd) {
        setTimeout(fn, 50);
      }
    };
    fn();
  }
}

export function wrapTextPreserveWord(s: string, maxWidth: number, fontSize: number) {
  let width = G6.Util.getTextSize(s, fontSize)[0];
  if (width <= maxWidth) {
    return s;
  }

  let words = s.split(" ");
  let lines: string[][] = [[]];
  let currentWidth = 0;

  for (let word of words) {
    let wordWidth = G6.Util.getTextSize(word, fontSize)[0];
    if (currentWidth + wordWidth > maxWidth) {
      lines.push([]);
      currentWidth = wordWidth;
    } else {
      currentWidth += wordWidth;
    }

    lines[lines.length - 1].push(word);
  }

  if (lines[lines.length - 1].length === 0) {
    lines.pop();
  }

  return lines.map((ws) => ws.join(" ")).join("\n");
}