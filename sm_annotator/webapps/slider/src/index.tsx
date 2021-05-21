import { Provider } from "mobx-react";
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import {
  AppWrapper,
  RecordTunnel,
  ReplayTunnel,
  Socket,
  Tunnel,
} from "./library";
import { AppStore } from "./models";

let Slider: any = {};
(window as any).Slider = Slider;

// exposing the function to call it from outside
Slider.renderApp = (
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
      <AppWrapper socket={socket} store={store as any} App={App} />
    </Provider>,
    shadow
  );
}

Slider.renderDevApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  let recordTunnel = new RecordTunnel(tunnel);
  Slider.recordTunnel = recordTunnel;
  console.log("renderDev");
  Slider.renderApp(containerId, recordTunnel, defaultProps, shadowDOM);
};

// render the app when debugging
if (process.env.REACT_APP_DEV === "yes") {
  let hist = require("./replayDebugData").history;
  Slider.renderApp("root", new ReplayTunnel(hist), { min: 0, max: 10, index: 0 });
}