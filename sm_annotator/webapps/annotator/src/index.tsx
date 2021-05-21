import { Provider } from "mobx-react";
import React from "react";
import ReactDOM from "react-dom";
import { THEME } from "./env";
import {
  AppWrapper,
  RecordTunnel,
  ReplayTunnel,
  Socket,
  Tunnel,
} from "./library";
import { AppStore } from "./models";
import App from "./App";
import VizSemModelsApp from "./VizSemModelsApp";

let Annotator: any = {};
(window as any).Annotator = Annotator;
// exposing the application for people to call it from outside
Annotator.renderApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  if (defaultProps === undefined) {
    defaultProps = {};
  }

  let container = document.getElementById(containerId);
  if (container === null) {
    console.error("Invalid container id");
    return;
  }

  let enableLogging = true;
  let shadow = shadowDOM === false ? container : container.attachShadow({ mode: "open" });
  let socket = new Socket(tunnel, 600000, enableLogging);
  let store = new AppStore(socket, defaultProps as any);

  store.setProps({ root: shadow });

  ReactDOM.render(
    <Provider store={store}>
      <AppWrapper socket={socket} store={store as any} App={App} />
    </Provider>,
    shadow
  );
}

Annotator.renderDevApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  let recordTunnel = new RecordTunnel(tunnel);
  Annotator.recordTunnel = recordTunnel;
  console.log("renderDev");
  Annotator.renderApp(containerId, recordTunnel, defaultProps, shadowDOM);
};

Annotator.renderVizSemModelApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  if (defaultProps === undefined) {
    defaultProps = {};
  }

  let container = document.getElementById(containerId);
  if (container === null) {
    console.error("Invalid container id");
    return;
  }

  let enableLogging = true;
  let shadow = shadowDOM === false ? container : container.attachShadow({ mode: "open" });
  let socket = new Socket(tunnel, 60000, enableLogging);
  let store = new AppStore(socket, defaultProps as any);

  store.setProps({ root: shadow });

  ReactDOM.render(
    <Provider store={store}>
      <AppWrapper socket={socket} store={store as any} App={VizSemModelsApp} />
    </Provider>,
    shadow
  );
}

Annotator.renderDevVizSemModelApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  let recordTunnel = new RecordTunnel(tunnel);
  Annotator.recordTunnel = recordTunnel;
  console.log("renderDev");
  Annotator.renderVizSemModelApp(containerId, recordTunnel, defaultProps, shadowDOM);
};

if (process.env.REACT_APP_DEV === "yes") {
  if (THEME === "dark") {
    (document.body as any).style = "background: black";
  }
  let hist = require("./replayDebugData").history;
  // let shadowDOM = true;
  let shadowDOM = false;
  Annotator.renderApp("root", new ReplayTunnel(hist), undefined, shadowDOM);
}
