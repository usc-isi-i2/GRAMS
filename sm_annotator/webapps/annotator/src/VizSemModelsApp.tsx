import { Button, Col, message, Row } from "antd";
import { observer } from "mobx-react";
import React from "react";
import SemanticModel from "./components/SemanticModel";
import { THEME, THEME_CSS } from "./env";
import { Socket } from "./library";
import { AppStore } from "./models";
import Table from "./components/Table";

interface Props {
  socket: Socket;
  store: AppStore;
}

interface State {
}

@observer
export default class VizSemModelsApp extends React.Component<Props, State> {
  private container = React.createRef<HTMLDivElement>();
  private smContainer1 = React.createRef<SemanticModel>();
  private smContainer2 = React.createRef<SemanticModel>();

  centerGraph = () => {
    if (this.smContainer1.current !== null) {
      this.smContainer1.current.centerGraph();
    }
    if (this.smContainer2.current !== null) {
      this.smContainer2.current.centerGraph();
    }
  };

  tableFetchData = () => {
    let table = this.props.store.props.table!;
    return table.updatePagination(table.pagination);
  };

  componentDidMount = () => {
    message.config({
      getContainer: () => this.container.current!,
    })
  }

  render() {
    let store = this.props.store;
    if (store.props.graphs === undefined || store.props.graphs.length === 0) {
      return <div></div>;
    }

    let twoColumnLayout = (store.props as any).twoColumnLayout;
    let component = null;
    if (twoColumnLayout) {
      component = <Row gutter={8}>
        <Col span={12}>
          <SemanticModel ref={this.smContainer1} graph={store.props.graphs[0]} disableEdit={true}/>
        </Col>
        <Col span={12}>
          <SemanticModel ref={this.smContainer2} graph={store.props.graphs[1]} disableEdit={true}/>
        </Col>
      </Row>
    } else {
      component = <React.Fragment>
        <SemanticModel ref={this.smContainer1} graph={store.props.graphs[0]} disableEdit={true}/>
        <SemanticModel ref={this.smContainer2} graph={store.props.graphs[1]} disableEdit={true}/>
      </React.Fragment>
    }
    return (
      <div ref={this.container} className={`app-${THEME}`}>
        <style type="text/css">{THEME_CSS}</style>
        <div className="mb-2">
          <Button size="small" className="mr-2" onClick={this.centerGraph}>
            Center graph
          </Button>
          <b>Note:</b> <span dangerouslySetInnerHTML={{__html: this.props.store.props.log.note}}></span>
        </div>
        {component}
        <Table />
      </div>
    );
  }
}
