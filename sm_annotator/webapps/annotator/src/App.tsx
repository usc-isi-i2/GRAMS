import { Button, Input, message, Spin, Switch } from "antd";
import { observer } from "mobx-react";
import React from "react";
import SemanticModel from "./components/SemanticModel";
import Table from "./components/Table";
import { THEME, THEME_CSS } from "./env";
import { Socket } from "./library";
import { AppStore, TableColumnRelationFilter } from "./models";
import { RecordSelection, Record, defaultRankByName } from "./components/primitives/RecordSelection";
import memoizeOne from "memoize-one";

interface Props {
  socket: Socket;
  store: AppStore;
}

interface State {
  loading: boolean;
}

@observer
export default class App extends React.Component<Props, State> {
  private container = React.createRef<HTMLDivElement>();
  private smContainer = React.createRef<SemanticModel>();
  public state: State = { loading: false };

  get table() {
    return this.props.store.props.table!;
  }

  getSelectSM = memoizeOne((nGraphs: number) => {
    let indice = Array.from({ length: nGraphs }, (v, i) => ({ id: i.toString(), label: i.toString() }));
    indice.push({ id: "clone", label: "Copy from current" })
    return indice;
  });

  selectSM = (r: Record) => {
    if (r.id === "clone") {
      // add new graph
      this.props.store.createNewGraphFromCurrentGraph();
    } else {
      this.props.store.setCurrentGraph(parseInt(r.id));
    }
  }

  removeCurrentSM = () => {
    this.props.store.removeCurrentGraph();
  }

  tableFetchData = () => {
    return this.table.updatePagination(this.table.pagination);
  };

  clearFilter = () => {
    this.table.updateFilter({
      columnTypeFilters: {},
      columnRelationFilters: new TableColumnRelationFilter({}),
      columnLinkFilters: {},
    });
  };

  centerGraph = () => {
    if (this.smContainer.current === null) {
      return;
    }
    this.smContainer.current.centerGraph();
  };

  showAddEdgeForm = () => {
    if (this.smContainer.current === null) {
      return;
    }
    this.smContainer.current.showAddEdgeForm();
  };

  showAddNodeForm = () => {
    if (this.smContainer.current === null) {
      return;
    }
    this.smContainer.current.showAddNodeForm();
  }

  saveSemanticModel = () => {
    this.setState({ loading: true });
    return this.props.store.save().then(() => {
      this.setState({ loading: false });
    });
  };

  componentDidMount = () => {
    message.config({
      getContainer: () => this.container.current!,
    })
  }

  render() {
    let store = this.props.store;
    if (
      store.props.table === undefined ||
      store.props.graphs.length === 0
    ) {
      return <div>Annotator can't start because no table are given or empty graph</div>;
    }

    let saveSMBtn;
    if (this.state.loading) {
      saveSMBtn = <Spin />;
    } else {
      saveSMBtn = (
        <Button
          size="small"
          type="primary"
          disabled={!store.stale}
          onClick={this.saveSemanticModel}
        >
          Save the work
        </Button>
      );
    }

    return (
      <div ref={this.container} className={`app-${THEME}`}>
        <style type="text/css">{THEME_CSS}</style>
        <div className="mb-2">
          {saveSMBtn}
          <span className="ml-2 mr-2">
            Current SM:
          </span>
          <RecordSelection
            selectRecord={{ id: store.props.currentGraphIndex.toString(), label: store.props.currentGraphIndex.toString() }}
            records={this.getSelectSM(store.props.graphs.length)}
            onSelectRecord={this.selectSM}
            rankByName={defaultRankByName(this.getSelectSM(store.props.graphs.length))}
            style={{ width: 'auto', display: 'inline-block' }}
          />
          <Button
            size="small"
            type="primary"
            danger={true}
            className="ml-2"
            disabled={store.props.graphs.length <= 1}
            onClick={this.removeCurrentSM}
          >
            Remove current SM
          </Button>
          <Button
            size="small"
            type="primary"
            className="ml-2"
            disabled={!this.table.hasFilters()}
            onClick={this.clearFilter}
          >
            Clear filters
          </Button>
          <Button size="small" className="ml-2" onClick={this.centerGraph}>
            Center graph
          </Button>
          <Button size="small" className="ml-2" onClick={this.showAddNodeForm}>
            Add node
          </Button>
          <Button size="small" className="ml-2" onClick={this.showAddEdgeForm}>
            Add edge
          </Button>
          <span className="ml-2">
            Is curated: <Switch checked={store.props.log.isCurated} onChange={(val) => store.updateIsCurated(val)} />
          </span>
          <span className="ml-2">Note:</span>
          <div className="mt-2">
            <Input value={store.props.log.note} onChange={(e) => store.updateNote(e.target.value)} />
          </div>
        </div>
        <SemanticModel ref={this.smContainer} graph={this.props.store.currentGraph} />
        <Table />
      </div>
    );
  }
}
