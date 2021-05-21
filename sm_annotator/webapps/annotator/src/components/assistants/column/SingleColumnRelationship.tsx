import { Switch } from "antd";
import React from "react";
import { Table, TableFilter, TableColumnRelationFilter, Resource } from "../../../models";
import { Relationship } from "../../../models/Assistant";
import { ExternalLink } from "../../primitives/ExternalLink";
import memoizeOne from 'memoize-one';


interface Props {
  table: Table;
  columnId: number;
  properties: [Resource, Resource][];
  updateFilter: (direction: "incoming" | "outgoing", endpoint: string | number, pred1: string, pred2: string, op: "include" | "exclude") => void;
  style?: React.CSSProperties;
  btnStyle?: React.CSSProperties;
}

interface State {
  display: boolean;
}

export class SingleColumnRelationshipComponent extends React.Component<Props, State> {
  public state: State = { display: false }

  toggleDisplay = () => {
    this.setState({ display: !this.state.display });
  }

  render() {
    let { table, properties, columnId } = this.props;
    let columnRelationFilter = table.filters.columnRelationFilters;

    let rows = null;
    if (this.state.display) {
      rows = properties.map(([prop, qual], index) => {
        let includeSwitch = false;
        let excludeSwitch = false;

        switch (columnRelationFilter.getFilterOp(columnId, "outgoing", TableColumnRelationFilter.WILDCARD_ENDPOINT, prop.uri, qual.uri)) {
          case "exclude":
            excludeSwitch = true;
            break;
          case "include":
            includeSwitch = true;
            break
        }

        return <tr key={index}>
          <td colSpan={prop.uri === qual.uri ? 2 : 1}>
            <ExternalLink url={prop.uri} style={{ textDecoration: "none" }}>
              {prop.label}
            </ExternalLink>
          </td>
          {prop.uri === qual.uri ? undefined : <td>
            <ExternalLink url={qual.uri} style={{ textDecoration: "none" }}>
              {qual.label}
            </ExternalLink>
          </td>}
          <td>
            <Switch
              size="small"
              checked={includeSwitch}
              onClick={() => this.props.updateFilter("outgoing", TableColumnRelationFilter.WILDCARD_ENDPOINT, prop.uri, qual.uri, "include")}
            />
          </td>
          <td>
            <Switch
              size="small"
              checked={excludeSwitch}
              onClick={() => this.props.updateFilter("outgoing", TableColumnRelationFilter.WILDCARD_ENDPOINT, prop.uri, qual.uri, "exclude")}
            />
          </td>
        </tr>
      });
    }

    return <table key="type-hierarchy" className="lightweight-table" style={this.props.style}>
      <thead>
        <tr>
          <th colSpan={4} style={{ textAlign: 'center', border: '2px solid #bbb', cursor: 'pointer', ...this.props.btnStyle }}
            onClick={this.toggleDisplay}>{this.state.display ? "Hide Properties" : "Show Properties"}</th>
        </tr>
        <tr style={{ display: this.state.display ? undefined : "none" }}>
          <th colSpan={2}>Property</th>
          <th>Include</th>
          <th>Exclude</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  }
}