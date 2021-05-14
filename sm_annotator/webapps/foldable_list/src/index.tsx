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

let FoldableList: any = {};
(window as any).FoldableList = FoldableList;

// exposing the function to call it from outside
FoldableList.renderApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  if (defaultProps === undefined) {
    defaultProps = { items: [] };
  }

  let container = document.getElementById(containerId);
  if (container === null) {
    console.error(`Invalid container id: ${containerId}`, document.getElementById(containerId));
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

FoldableList.renderDevApp = (
  containerId: string,
  tunnel: Tunnel,
  defaultProps?: { [prop: string]: any },
  shadowDOM?: boolean
) => {
  let recordTunnel = new RecordTunnel(tunnel);
  FoldableList.recordTunnel = recordTunnel;
  console.log("renderDev");
  FoldableList.renderApp(containerId, recordTunnel, defaultProps, shadowDOM);
};

// render the app when debugging
if (process.env.REACT_APP_DEV === "yes") {
  let hist = require("./replayDebugData").history;
  let defaultProps = {
    header: "city in us",
    items: [
      "<b>human (Q5)</b>",
      {
        header: "US (Q30)",
        items: [
          {
            header: "US (Q30)",
            items: [
              {
                header: "US (Q30)",
                items: [
                  "located (P131)",
                  "area (P587)"
                ]
              },
              "located (P131)",
              "area (P587)"
            ]
          },
          "located (P131)",
          "area (P587)",

        ]
      },
      {
        header: "USA (Q30)",
        items: [
        ]
      }
    ]
  };
  FoldableList.renderApp("root", new ReplayTunnel(hist), defaultProps);
}