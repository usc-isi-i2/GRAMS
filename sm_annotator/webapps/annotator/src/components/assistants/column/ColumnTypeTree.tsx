import { Button, Switch } from "antd";
import {
  EyeInvisibleOutlined
} from '@ant-design/icons';
import React from "react";
import { TableFilter } from "../../../models";
import { TypeHierarchy } from "../../../models/Assistant";
import { ExternalLink } from "../../primitives/ExternalLink";


interface Props {
  filters: TableFilter;
  columnIndex: number;
  flattenTypeHierarchy: TypeHierarchy[];
  updateFilter: (uri: string, op: "include" | "exclude") => void;
  selectNEType: (type: TypeHierarchy) => void;
  style?: React.CSSProperties;
  btnStyle?: React.CSSProperties;
}

interface State {
  showTypes: boolean[];
  showTable: boolean;
}

export function getIndentIndicator(depth: number) {
  let indent = [];
  for (let i = 0; i < depth; i++) {
    if (i === 0) {
      indent.push("\u00B7".repeat(3));
    } else {
      indent.push(<span key={i} style={{ display: 'inline-block', transform: 'scale(1, 4)' }}>{"\uff5c"}</span>);
      indent.push("\u00B7".repeat(3));
    }
  }
  indent.push("↳");
  return <span style={{ fontFamily: "monospace", fontSize: 10 }}>{indent}</span>
}

export class TypeTree extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      showTypes: this.getShowTypes(props, this.getHeuristicMinPercentage(props.flattenTypeHierarchy)),
      showTable: false,
    };
  }

  UNSAFE_componentWillReceiveProps(nextProps: Props) {
    if (nextProps.flattenTypeHierarchy !== this.props.flattenTypeHierarchy) {
      // check this so that updating the filter don't reset the showTypes
      this.setState({ showTypes: this.getShowTypes(nextProps, this.getHeuristicMinPercentage(nextProps.flattenTypeHierarchy)) });
    }
  }

  getHeuristicMinPercentage = (types: TypeHierarchy[]) => {
    if (types.length < 2) {
      return 0.0;
    }

    // return the second highest freq.
    let freqs = Array.from(new Set(types.map((type) => type.freq))).sort();
    if (freqs.length === 1) {
      return types[0].percentage - 1e-7;
    }

    return types[types.findIndex((type) => type.freq === freqs[freqs.length - 2])].percentage - 1e-7;
  }

  getShowTypes = (props: Props, minPercentage: number) => {
    let showTypes = [];
    for (let i = 0; i < props.flattenTypeHierarchy.length; i++) {
      showTypes.push(props.flattenTypeHierarchy[i].percentage >= minPercentage);
    }

    return showTypes;
  }

  showTree = (index: number) => {
    // click to show consecutive hidden types
    let ptr = index;
    let showTypes = this.state.showTypes;
    while (ptr > 0 && !showTypes[ptr]) {
      showTypes[ptr] = true;
      ptr -= 1;
    }
    ptr = index + 1;
    while (ptr < showTypes.length && !showTypes[ptr]) {
      showTypes[ptr] = true;
      ptr += 1;
    }

    this.setState({ showTypes });
  }

  hideTree = (index: number) => {
    // hide the current type and its children
    let showTypes = this.state.showTypes;
    showTypes[index] = false;
    let currentDepth = this.props.flattenTypeHierarchy[index].depth
    index += 1;
    while (index < this.props.flattenTypeHierarchy.length) {
      if (this.props.flattenTypeHierarchy[index].depth > currentDepth) {
        showTypes[index] = false;
        index += 1;
      } else {
        break;
      }
    }
    this.setState({ showTypes });
  }

  toggleDisplay = () => {
    // toggle displaying of this component, on hide it looks like a button
    this.setState({ showTable: !this.state.showTable });
  }

  render() {
    let colFilter = this.props.filters.columnTypeFilters[this.props.columnIndex] || {};
    let typeComponents = null;

    if (this.state.showTable) {
      typeComponents = this.props.flattenTypeHierarchy
        .map((type, index) => [type, index] as [TypeHierarchy, number])
        .filter(([type, index]) => {
          if (index > 0 && this.state.showTypes[index] === false && this.state.showTypes[index - 1] === false) {
            return false;
          }
          return true;
        })
        .map(([type, index]) => {
          if (this.state.showTypes[index] === false) {
            return <tr key={index}>
              <td colSpan={6} style={{ paddingTop: 8, paddingBottom: 2, cursor: "pointer" }} onClick={() => this.showTree(index)}><p style={{ color: '#1890ff', fontSize: "0.5em", textAlign: "center" }}>&#9679;&#9679;&#9679;</p></td>
            </tr>
          }

          let prefix = type.depth === 0 ? "" : getIndentIndicator(type.depth);
          let includeSwitch = false;
          let excludeSwitch = false;

          switch (colFilter[type.uri]) {
            case "include":
              includeSwitch = true;
              break;
            case "exclude":
              excludeSwitch = true;
              break;
          }

          return <tr key={index}>
            <td>{type.freq}</td>
            <td><EyeInvisibleOutlined onClick={() => this.hideTree(index)} /></td>
            <td>
              <ExternalLink url={type.uri} style={{ textDecoration: "none" }}>
                {prefix} ({Math.round(type.percentage * 100)}%) {type.label}
                {type.duplicated ? " (duplicated)" : ""}
              </ExternalLink>
            </td>
            <td>
              <Switch
                size="small"
                checked={includeSwitch}
                onClick={() => this.props.updateFilter(type.uri, "include")}
              />
            </td>
            <td>
              <Switch
                size="small"
                checked={excludeSwitch}
                onClick={() => this.props.updateFilter(type.uri, "exclude")}
              />
            </td>
            <td>
              <Button size="small" onClick={() => this.props.selectNEType(type)}>Select</Button>
            </td>
          </tr>
        });
    }

    return <table key="type-hierarchy" className="lightweight-table" style={this.props.style}>
      <thead>
        <tr>
          <th colSpan={6} style={{ textAlign: 'center', border: '2px solid #bbb', cursor: 'pointer', ...this.props.btnStyle }}
            onClick={this.toggleDisplay}>{this.state.showTable ? "Hide Types" : "Show Types"}</th>
        </tr>
        <tr style={{ display: this.state.showTable ? undefined : "none" }}>
          <th>#</th>
          <th>Hide</th>
          <th>Class</th>
          <th>Include</th>
          <th>Exclude</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>
        {typeComponents}
      </tbody>
    </table>
  }
}