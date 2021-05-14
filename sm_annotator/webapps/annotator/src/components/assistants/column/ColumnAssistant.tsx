import { message, Spin, Switch } from "antd";
import { inject, observer } from "mobx-react";
import React from "react";
import { Table, Graph, AppStore, Resource } from "../../../models";
import { Assistant, Relationship } from "../../../models/Assistant";
import { TypeTree } from "./ColumnTypeTree";
import { RelationshipComponent } from "./InterColumnRelationship";
import { SingleColumnRelationshipComponent } from "./SingleColumnRelationship";
import memoizeOne from "memoize-one";


interface Props {
  table?: Table;
  graph?: Graph;
  assistant?: Assistant;
  columnId: number;
}

interface State { }

@inject((provider: { store: AppStore }) => ({
  graph: provider.store.currentGraph,
  table: provider.store.props.table,
  assistant: provider.store.props.assistant,
}))
@observer
export default class ColumnAssistantComponent extends React.Component<Props, State> {
  get suggestions() {
    return this.props.assistant!.columns[this.props.columnId];
  }

  componentDidMount() {
    // send out query to get some assistant
    if (this.props.assistant!.columns[this.props.columnId] === undefined) {
      this.props.assistant!.requestColumnSuggestions(this.props.columnId);
    }
  }

  selectNEType = (classType: Resource) => {
    // select NE type
    try {
      this.props.graph!.upsertColumnType(this.props.columnId, {
        uri: classType.uri,
        label: classType.label,
        approximation: false,
      });
    } catch (error) {
      message.error(error.message);
      console.error(error);
    }
  }

  selectRelationship = (sourceEndpoint: string, targetEndpoint: string, pred1: string, pred2: string) => {
    // if the source id doesn't exist, we need to create it too
    let sourceId, targetId;
    // try {
    //   this.props.graph!.upsertColumnType(this.props.columnId, {
    //     uri: classType.uri,
    //     label: classType.label,
    //     approximation: false,
    //   });
    // } catch (error) {
    //   message.error(error.message);
    //   console.error(error);
    // }
  }

  updateTypeFilter = (qnode_id: string, op: "include" | "exclude") => {
    // update the filter
    let columnId = this.props.columnId;
    let filters = JSON.parse(
      JSON.stringify(this.props.table!.filters.columnTypeFilters)
    );
    if (filters[columnId] === undefined) {
      filters[columnId] = {};
    }

    if (filters[columnId][qnode_id] !== op) {
      // set the op
      filters[columnId][qnode_id] = op;
      for (let child of this.suggestions!.type2children![qnode_id]) {
        filters[columnId][child] = op;
      }
    } else {
      // toggle it
      delete filters[columnId][qnode_id];
      for (let child of this.suggestions!.type2children![qnode_id]) {
        delete filters[columnId][child];
      }
      if (Object.keys(filters[columnId]).length === 0) {
        delete filters[columnId];
      }
    }

    this.props.table!.updateFilter({
      ...this.props.table!.filters,
      columnTypeFilters: filters,
    });
  }

  updateRelationFilter = (direction: "incoming" | "outgoing", endpoint: number | string, pred1: string, pred2: string, op: "include" | "exclude") => {
    // update the filter
    let columnId = this.props.columnId;
    let filters = this.props.table!.filters.columnRelationFilters.shallowClone();
    let currentOp = filters.getFilterOp(columnId, direction, endpoint, pred1, pred2);
    switch (currentOp) {
      case undefined:
        filters.setFilterOp(columnId, direction, endpoint, pred1, pred2, op);
        break;
      default:
        if (currentOp === op) {
          filters.removeFilterOp(columnId, direction, endpoint, pred1, pred2);
        } else {
          filters.setFilterOp(columnId, direction, endpoint, pred1, pred2, op);
        }
        break;
    }
    this.props.table!.updateFilter({
      ...this.props.table!.filters,
      columnRelationFilters: filters,
    });
  }

  updateLinkFilter = (type: "hasLink" | "hasEntity" | "noLink" | "noEntity" | "none") => {
    let columnId = this.props.columnId;
    let linkFilter = Object.assign({}, this.props.table!.filters.columnLinkFilters);
    if (type === "none") {
      delete linkFilter[columnId];
    } else {
      linkFilter[columnId] = type;
    }
    this.props.table!.updateFilter({
      ...this.props.table!.filters,
      columnLinkFilters: linkFilter
    });
  }

  getColumnProperties = (suggestedRelationships: Relationship[]): [Resource, Resource][] => {
    let props: { [k: string]: [Resource, Resource] } = {};
    for (let rel of suggestedRelationships) {
      let key = JSON.stringify([rel.predicates[0].uri, rel.predicates[1].uri]);
      props[key] = rel.predicates;
    }
    // get relationships specified in the graph
    let graph = this.props.graph!;
    let classid = graph.getClassIdOfColumnId(this.props.columnId);
    if (classid !== undefined) {
      for (let [pe, qe] of graph.getOutgoingProperties(classid)) {
        if (qe === undefined) {
          qe = pe;
        }
        let key = JSON.stringify([pe.uri, qe.uri]);
        props[key] = [
          { uri: pe.uri, label: pe.label },
          { uri: qe.uri, label: qe.label },
        ]
      }
    }
    return Object.values(props);
  };

  render() {
    let table = this.props.table!;
    let columnId = this.props.columnId;
    let assistant = this.props.assistant!;
    let suggestions = this.suggestions;
    if (suggestions === undefined) {
      return null;
    }

    if (assistant.loading) {
      return <div style={{ textAlign: 'center' }}><Spin /></div>
    }
    let components = [];
    if (suggestions.stats !== undefined && Object.keys(suggestions.stats).length > 0) {
      let keys = [];
      let values = [];

      for (let [k, v] of Object.entries(suggestions.stats)) {
        keys.push(<th key={k}>{k}</th>);
        values.push(<td key={k}>{v}</td>);
      }

      components.push(<table key='horizontal-stat' className="lightweight-table" style={{ width: '100%' }}>
        <thead>
          <tr>
            {keys}
            <th>Has Link</th>
            <th>Has No Link</th>
            <th>Has Entity</th>
            <th>Has No Entity</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            {values}
            <td>
              <Switch
                size="small"
                checked={table.filters.columnLinkFilters[columnId] === "hasLink"}
                onClick={() => this.updateLinkFilter(table.filters.columnLinkFilters[columnId] === "hasLink" ? "none" : "hasLink")}
              />
            </td>
            <td>
              <Switch
                size="small"
                checked={table.filters.columnLinkFilters[columnId] === "noLink"}
                onClick={() => this.updateLinkFilter(table.filters.columnLinkFilters[columnId] === "noLink" ? "none" : "noLink")}
              />
            </td>
            <td>
              <Switch
                size="small"
                checked={table.filters.columnLinkFilters[columnId] === "hasEntity"}
                onClick={() => this.updateLinkFilter(table.filters.columnLinkFilters[columnId] === "hasEntity" ? "none" : "hasEntity")}
              />
            </td>
            <td>
              <Switch
                size="small"
                checked={table.filters.columnLinkFilters[columnId] === "noEntity"}
                onClick={() => this.updateLinkFilter(table.filters.columnLinkFilters[columnId] === "noEntity" ? "none" : "noEntity")}
              />
            </td>
          </tr>
        </tbody>
      </table>);
    }

    if (suggestions.flattenTypeHierarchy !== undefined) {
      components.push(<TypeTree
        key="type-hierarchy" filters={table.filters} style={{ width: '100%', marginTop: components.length > 0 ? 8 : 0 }} btnStyle={{ backgroundColor: '#d9f7be' }}
        columnIndex={columnId} flattenTypeHierarchy={suggestions.flattenTypeHierarchy}
        updateFilter={this.updateTypeFilter} selectNEType={this.selectNEType} />);
    }

    if (suggestions.relationships !== undefined) {
      components.push(<RelationshipComponent key="relationship-incoming"
        style={{ width: '100%', marginTop: components.length > 0 ? 8 : 0 }} btnStyle={{ backgroundColor: '#bae7ff' }}
        direction="incoming" relationships={suggestions.relationships.incoming}
        columnId={columnId} table={table}
        updateFilter={this.updateRelationFilter}
      />);
      components.push(<RelationshipComponent key="relationship-outgoing"
        style={{ width: '100%', marginTop: components.length > 0 ? 8 : 0 }} btnStyle={{ backgroundColor: '#fff1b8' }}
        direction="outgoing" relationships={suggestions.relationships.outgoing}
        columnId={columnId} table={table}
        updateFilter={this.updateRelationFilter}
      />);
      components.push(<SingleColumnRelationshipComponent key="property-outgoing"
        style={{ width: '100%', marginTop: components.length > 0 ? 8 : 0 }} btnStyle={{ backgroundColor: '#efdbff' }}
        properties={this.getColumnProperties(suggestions.relationships.outgoing)}
        columnId={columnId} table={table}
        updateFilter={this.updateRelationFilter}
      />);
    }

    return <React.Fragment>{components}</React.Fragment>
  }
}