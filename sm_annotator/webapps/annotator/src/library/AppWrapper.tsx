import React from "react";
import { observer } from "mobx-react";
import { Store } from "./Store";
import { Socket } from "./Socket";

interface Props {
  socket: Socket;
  store: Store;
  App: typeof React.Component
}

@observer
export default class AppWrapper extends React.Component<Props> {
  public static defaultProps = { requestTimeoutMs: 500 };
  private app = React.createRef<any>();
  private pendingOps: { version: number; resolve: () => void }[] = [];

  constructor(props: Props) {
    super(props);

    // on set props
    this.props.socket.onSetProps((newProps: { [prop: string]: any }) => {
      return new Promise((resolve) => {
        let version = this.props.store.version + 1;
        this.pendingOps.push({ version, resolve });
        this.props.store.setProps(newProps, version);
      });
    });

    // on reset props
    this.props.socket.onResetProps(() => {
      return new Promise((resolve) => {
        let version = this.props.store.version + 1;
        this.pendingOps.push({ version, resolve });
        this.props.store.resetProps(version);
      });
    });
    // on execution function
    this.props.socket.onExecFunc((func: string, args: any[]) => {
      let resp;

      if (func.startsWith("store.")) {
        resp = (this.props.store as any)[func.substring('store.'.length)].apply(this.props.store, args);
      } else {

        if (this.app.current === null) {
          console.error(
            `Cannot call function ${func} at the current state as the application is not finished rendering`
          );
          return Promise.reject("Cannot call this function at the current state");
        }
        resp = (this.app.current as any)![func.substring("app.".length)].apply(this.app.current, args);
      }

      return resp !== undefined && resp.then !== undefined
        ? resp
        : Promise.resolve(resp);
    });
  }

  componentDidUpdate() {
    while (
      this.pendingOps.length > 0 &&
      this.props.store.version >= this.pendingOps[0].version
    ) {
      this.pendingOps.shift()!.resolve();
    }
  }

  componentDidMount() {
    // send a message to the otherside telling that we have finished starting.
    this.props.socket.isClientReady = true;
  }

  render() {
    let App = this.props.App;
    // need a hidden element to store the version, otherwise, it won't be trigger
    // when the store is changing
    return (
      <div>
        <div style={{ display: "none" }}>{this.props.store.version}</div>
        <App ref={this.app} socket={this.props.socket} store={this.props.store as any} />
      </div>
    );
  }
}
