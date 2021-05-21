import { Button, Switch } from "antd";
import {
  EyeInvisibleOutlined
} from '@ant-design/icons';
import React from "react";
import { Table, TableFilter } from "../../../models";
import { ExternalLink } from "../../primitives/ExternalLink";
import { TypeHierarchy, Relationship } from "../../../models/Assistant";
import memoizeOne from 'memoize-one';
import { observer } from "mobx-react";

interface Props {
  direction: "incoming" | "outgoing";
  table: Table;
  columnId: number;
  updateFilter: (direction: "incoming" | "outgoing", endpoint: string | number, pred1: string, pred2: string, op: "include" | "exclude") => void;
  relationships: Relationship[];
  style?: React.CSSProperties;
  btnStyle?: React.CSSProperties;
}

interface State {
  display: boolean;
}

@observer
export class RelationshipComponent extends React.Component<Props, State> {
  public state: State = { display: false }

  toggleDisplay = () => {
    this.setState({ display: !this.state.display });
  }

  getRelationships = memoizeOne((rels: Relationship[]) => {
    let endpoint2rels: { [x: string]: Relationship[] } = {};
    for (let rel of rels) {
      if (typeof rel.endpoint === "number") {
        if (endpoint2rels[rel.endpoint] === undefined) {
          endpoint2rels[rel.endpoint] = [];
        }
        endpoint2rels[rel.endpoint].push(rel);
      } else {
        if (endpoint2rels[rel.endpoint.uri] === undefined) {
          endpoint2rels[rel.endpoint.uri] = [];
        }
        endpoint2rels[rel.endpoint.uri].push(rel);
      }
    }

    return Object.values(endpoint2rels);
  });

  render() {
    let { direction, table, relationships, columnId } = this.props;
    let columnRelationFilter = table.filters.columnRelationFilters;

    let rows = null;
    if (this.state.display) {
      rows = this.getRelationships(relationships).flatMap((rels, gindex) => {
        return rels.map((rel, index) => {
          let label, endpoint: string | number;
          if (typeof rel.endpoint === "number") {
            // column
            label = `${table.getColumnById(rel.endpoint).title} (${rel.endpoint})`;
            endpoint = rel.endpoint;
          } else {
            label = rel.endpoint.label;
            endpoint = rel.endpoint.uri;
          }

          let includeSwitch = false;
          let excludeSwitch = false;
          let propComponents;

          switch (columnRelationFilter.getFilterOp(columnId, direction, endpoint, rel.predicates[0].uri, rel.predicates[1].uri)) {
            case "exclude":
              excludeSwitch = true;
              break;
            case "include":
              includeSwitch = true;
              break
          }

          if (rel.predicates[0].uri === rel.predicates[1].uri) {
            propComponents = <td className={index === rels.length - 1 ? "row-separator" : ""} colSpan={2}>
              <ExternalLink url={rel.predicates[0].uri} style={{ textDecoration: "none" }}>
                {rel.predicates[0].label}
              </ExternalLink>
            </td>
          } else {
            propComponents = <React.Fragment>
              <td className={index === rels.length - 1 ? "row-separator" : ""}>
                <ExternalLink url={rel.predicates[0].uri} style={{ textDecoration: "none" }}>
                  {rel.predicates[0].label}
                </ExternalLink>
              </td>
              <td className={index === rels.length - 1 ? "row-separator" : ""}>
                <ExternalLink url={rel.predicates[1].uri} style={{ textDecoration: "none" }}>
                  {rel.predicates[1].label}
                </ExternalLink>
              </td>
            </React.Fragment>
          }

          return <tr key={`${gindex}-${index}`}>
            {index === 0 ? <td rowSpan={rels.length} className="row-separator">
              {label}
            </td> : null}
            {propComponents}
            <td className={index === rels.length - 1 ? "row-separator" : ""}>{rel.freq}</td>
            <td className={index === rels.length - 1 ? "row-separator" : ""}>
              <Switch
                size="small"
                checked={includeSwitch}
                onClick={() => this.props.updateFilter(direction, endpoint, rel.predicates[0].uri, rel.predicates[1].uri, "include")}
              />
            </td>
            <td className={index === rels.length - 1 ? "row-separator" : ""}>
              <Switch
                size="small"
                checked={excludeSwitch}
                onClick={() => this.props.updateFilter(direction, endpoint, rel.predicates[0].uri, rel.predicates[1].uri, "exclude")}
              />
            </td>
            <td className={index === rels.length - 1 ? "row-separator" : ""}>
              <Button size="small" disabled={true}>Select</Button>
            </td>
          </tr>
        })
      });
    }

    return <table className="lightweight-table" style={this.props.style}>
      <thead>
        <tr>
          <th colSpan={7} style={{ textAlign: 'center', border: '2px solid #bbb', cursor: 'pointer', ...this.props.btnStyle }} onClick={this.toggleDisplay}>
            {this.state.display ? `Hide ${direction} links` : `Show ${direction} links`}
          </th>
        </tr>
        <tr style={{ display: this.state.display ? undefined : "none" }}>
          <th>{direction === "incoming" ? "Source" : "Target"}</th>
          <th>Property</th>
          <th>Qualifier</th>
          <th>Frequency</th>
          <th>Include</th>
          <th>Exclude</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  }
}