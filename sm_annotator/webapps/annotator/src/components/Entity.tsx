import React from "react";
import { toJS } from "mobx";
import { inject, observer } from "mobx-react";
import { Entity, EntityStore, AppStore, Resource } from "../models";
import { ExternalLink } from "./primitives/ExternalLink";
import { message, Row, Col } from "antd";
import {
  SearchOutlined, StopOutlined, LoadingOutlined, HighlightFilled, FilterFilled, CaretDownOutlined,
  CaretUpOutlined
} from "@ant-design/icons";
import ReactDOM from "react-dom";

interface Props {
  entity: Entity;
  highlightProps?: string[];
}

interface State {
  mode: "onlyHighlight" | "showAll";
  showAllStmtValue: { [prop: string]: boolean };
}

export class EntityComponent extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      // default to onlyHighlight when there the highlight props is supplied
      mode: props.highlightProps !== undefined ? "onlyHighlight" : "showAll",
      showAllStmtValue: {}
    }
  }

  public toggleViewMode = (mode: "onlyHighlight") => {
    if (this.state.mode === mode) {
      // default is to showAll according to the UI
      this.setState({ mode: "showAll" });
    } else {
      this.setState({ mode });
    }
  }

  public showPortionPropValue = (puri: string) => {
    this.setState({ showAllStmtValue: { ...this.state.showAllStmtValue, [puri]: false } });
  }

  public showAllPropValue = (puri: string) => {
    this.setState({ showAllStmtValue: { ...this.state.showAllStmtValue, [puri]: true } });
  }

  render() {
    let ent = this.props.entity;
    let propComponents = [];
    let props = [];
    let needHighlightProps = new Set();

    let highlightColor = '#f5222d';
    let topKStmt = 5;

    if (this.state.mode === "onlyHighlight") {
      props = (this.props.highlightProps || [])
        .filter(p => ent.props[p] !== undefined)
        .map(p => ent.props[p]);
    } else {
      props = Object.values(ent.props);
      needHighlightProps = new Set(this.props.highlightProps || []);
    }

    for (let pval of props) {
      let stmts = [];
      let stmtLength = this.state.showAllStmtValue[pval.uri] === true ? pval.values.length : Math.min(topKStmt, pval.values.length);
      for (let i = 0; i < stmtLength; i++) {
        let stmt = pval.values[i];
        stmts.push(
          <div key={`stmt-${i}`} style={{ paddingTop: stmts.length > 0 ? 16 : undefined }}>
            {typeof stmt.value === 'string' ?
              stmt.value :
              <ExternalLink url={stmt.value.uri}>{stmt.value.label}</ExternalLink>}
            <table style={{ marginLeft: 36 }}>
              <tbody>
                {Object.values(stmt.qualifiers).map(qual => {
                  return <tr key={qual.uri}>
                    <td>
                      <ExternalLink url={qual.uri}>{qual.label}</ExternalLink>
                    </td>
                    <td style={{ paddingLeft: 24 }}>
                      {qual.values.map((qval, qidx) => {
                        return <div key={qidx}>{typeof qval === 'string' ? qval : <ExternalLink url={qval.uri}>{qval.label}</ExternalLink>}</div>
                      })}
                    </td>
                  </tr>
                })}
              </tbody>
            </table>
          </div>
        );
      }

      let ctl = null;
      if (stmtLength < pval.values.length) {
        ctl = <a style={{ textDecoration: "none" }} onClick={() => this.showAllPropValue(pval.uri)}>
          <CaretDownOutlined /> Show more ({pval.values.length} values)
        </a>;
      } else if (stmtLength > topKStmt) {
        ctl = <a style={{ textDecoration: "none" }} onClick={() => this.showPortionPropValue(pval.uri)}>
          <CaretUpOutlined /> Hide some ({pval.values.length} values)
        </a>;
      }

      propComponents.push(<tr key={pval.uri}>
        <td style={{
          width: '20%', minWidth: 150, verticalAlign: 'top',
          backgroundColor: 'rgb(234, 236, 240)', border: '1px solid #c8ccd1', padding: 8
        }}
        >
          <h4>
            <ExternalLink url={pval.uri} style={{ color: needHighlightProps.has(pval.uri) ? highlightColor : undefined }}>
              {pval.label}
            </ExternalLink>
          </h4>
          <div key="ctl">{ctl}</div>
        </td>
        <td style={{ padding: 8, border: '1px solid #c8ccd1', borderLeft: 0, verticalAlign: 'top' }}>
          {stmts}
        </td>
      </tr>);
    }

    let filterColor = this.state.mode === "onlyHighlight" ? "#389e0d" : undefined;

    return <div>
      <div style={{ overflow: 'auto' }}>
        <h3 style={{ float: "left" }}>
          <ExternalLink url={ent.uri}>{ent.label}</ExternalLink>
        </h3>
        <span style={{ marginLeft: 8 }}>
          ({Object.keys(ent.props).length} properties)
        </span>
        <span style={{ marginLeft: 8, cursor: 'pointer', color: filterColor, fontWeight: 500 }} onClick={() => this.toggleViewMode("onlyHighlight")}>
          Filter <FilterFilled />
        </span>
      </div>
      <p>{ent.description}</p>
      <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 8px' }}>
        <tbody>
          {propComponents}
        </tbody>
      </table>
    </div>
  }
}

interface OpenerProps {
  entityURI: string;
  highlightProps?: string[];
  entities?: EntityStore;
  store?: AppStore;
  autoHighlight: boolean;
  // supply the open function if you want to override the new tab behaviour
  open?: (entity: Entity) => void;
}

@inject((provider: { store: AppStore }) => ({
  entities: provider.store.props.entities,
  store: provider.store
}))
@observer
export class EntityComponentExternalOpener extends React.Component<OpenerProps, {}> {
  get entity() {
    return this.props.entities!.getEntity(this.props.entityURI);
  }

  open = () => {
    let ent = this.entity!;
    if (this.props.open !== undefined) {
      return this.props.open(ent);
    }

    let win = window.open("about:blank", `Entity ${ent.label}`)!;
    let div = win.document.createElement("div");
    div.id = "root";
    win!.document.body.appendChild(div);

    let title = win.document.createElement("title");
    title.innerHTML = ent.label;
    win.document.head.appendChild(title);

    let css = `
      div#root {
        font-family: "SF Pro Text", sans-serif
      }
      h1, h2, h3, h4, h5, h6 {
        margin-top: 0;
        margin-bottom: .5em;
        color: rgba(0,0,0,.85);
        font-weight: 500;
      }
      a { 
        color: #1890ff;
        text-decoration: underline;
        background-color: transparent;
        outline: none;
        cursor: pointer;
        -webkit-transition: color .3s;
        transition: color .3s;
        -webkit-text-decoration-skip: objects;
      }
    `;
    let highlightProps = this.props.highlightProps;
    if (highlightProps === undefined && this.props.autoHighlight) {
      highlightProps = this.props.store!.relevantProps;
    }

    ReactDOM.render(
      <div>
        <style>{css}</style>
        <EntityComponent entity={ent} highlightProps={highlightProps} />
      </div>
      , div
    );
  }

  componentDidMount() {
    if (this.entity === undefined) {
      this.props.entities!.fetchData([this.props.entityURI]);
    }
  }

  render() {
    let style = { margin: 6, cursor: 'pointer' };
    let ent = this.entity;
    if (ent === null) {
      message.error(`Entity ${this.props.entityURI} does not exist!`);
      return <StopOutlined style={style} />;
    }

    if (ent === undefined) {
      // wait for it to load
      return <LoadingOutlined style={style} />;
    }

    return <SearchOutlined style={style} onClick={this.open} />
  }
}
