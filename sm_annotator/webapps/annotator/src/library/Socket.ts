import { Tunnel } from "./Tunnel";

type SetPropsCallback = (props: { [prop: string]: any }) => Promise<void>;
type ResetPropsCallback = () => Promise<void>;
type ExecFuncCallback = (func: string, args: any[]) => Promise<void>;

export class Socket {
  private tunnel: Tunnel;
  // assign each request an id so that we can match response
  private requestId: number;
  private onSetPropsCB: SetPropsCallback = () => Promise.resolve();
  private onResetPropsCB: ResetPropsCallback = () => Promise.resolve();
  private onExecFuncCB: ExecFuncCallback = () => Promise.resolve();
  private requests: Map<string, [number, any, any]> = new Map();
  private requestTimeoutMs: number;
  private enableLogging: boolean;
  public isClientReady: boolean = false;

  constructor(
    tunnel: Tunnel,
    requestTimeoutMs: number,
    enableLogging: boolean
  ) {
    this.enableLogging = enableLogging;
    this.tunnel = tunnel;
    this.requestId = 0;
    this.tunnel.on_receive(this.handleIncomingMsg);
    this.requestTimeoutMs = requestTimeoutMs;
  }

  public onSetProps = (cb: SetPropsCallback) => {
    this.onSetPropsCB = cb;
    return this;
  };

  public onResetProps = (cb: ResetPropsCallback) => {
    this.onResetPropsCB = cb;
    return this;
  };

  public onExecFunc = (cb: ExecFuncCallback) => {
    this.onExecFuncCB = cb;
    return this;
  };

  /**
   * Set a request and return a promise that will be fulfilled when
   * the response is available
   */
  public request = <T1, T2>(
    url: string,
    params: T1
  ): Promise<{
    requestVersion: number;
    responseVersion: number;
    response: T2;
  }> => {
    let requestId = this.requestId.toString();
    this.requestId += 1;

    return new Promise((resolve, reject) => {
      // console.log("send request", JSON.stringify({ type: "request", id: requestId, params: params }));
      // put them into the queue waiting for response
      this.logRoot(`Send a request ${requestId} with params`, params);
      let version = this.tunnel.send_msg(
        JSON.stringify({
          id: requestId,
          url: url,
          params: params,
        })
      );
      this.requests.set(requestId, [version, resolve, reject]);

      setTimeout(() => {
        if (this.requests.has(requestId)) {
          this.requests.delete(requestId);
          console.log("timeout", this.requestTimeoutMs);
          reject({
            message: "Request exceed timeout",
            requestVersion: version,
            type: "timeout",
          });
        }
      }, this.requestTimeoutMs);
    });
  };

  /**
   * Main entry that handle incoming messages from the tunnel
   * @param version
   * @param msg
   */
  private handleIncomingMsg = (version: number, msg: string) => {
    let payload = JSON.parse(msg);
    // handle single action
    if (!Array.isArray(payload)) {
      let promise = Promise.resolve();
      switch (payload.type) {
        case "set_props":
          if (this.enableLogging) {
            this.group(`Set store properties`);
            console.log("Update the following properties", payload.props);
          }
          promise = this.onSetPropsCB(payload.props);
          break;
        case "reset_props":
          if (this.enableLogging) {
            this.group(`Reset store properties`);
          }
          promise = this.onResetPropsCB();
          break;
        case "exec_func":
          if (this.enableLogging) {
            this.group(`Execute the function ${payload.func}`);
            console.log(`Arguments:`, payload.args);
          }
          promise = this.onExecFuncCB(payload.func, payload.args);
          break;
        case "response":
          this.handleResponse(version, payload);
          break;
        default:
          console.error("Invalid message type! Get:", payload.type);
      }
      if (this.enableLogging) {
        console.log(`Finish!`);
        console.groupEnd();
      }
      return promise;
    }

    // handle a list of actions
    let promise = Promise.resolve();
    if (this.enableLogging) {
      promise = promise.then(() => {
        this.group(`Receive ${payload.length} messages`);
      });
    }

    for (let item of payload) {
      switch (item.type) {
        case "wait_for_client_ready":
          let ensurePromise: Promise<void> = new Promise((resolve, reject) => {
            let waitFor = () => {
              if (this.isClientReady) {
                return resolve();
              }
              setTimeout(waitFor, 25);
            };
            waitFor();
          });

          if (this.enableLogging) {
            promise = promise.then(() => {
              this.group(`Wait for client ready`);
              console.log("Start");
            });
          }
          promise = promise.then(() => {
            return ensurePromise;
          });
          if (this.enableLogging) {
            promise = promise.then(() => {
              console.log("Finish!");
              console.groupEnd();
            });
          }
          break;
        case "set_props":
          if (this.enableLogging) {
            promise = promise.then(() => {
              this.group(`Set store properties`);
              console.log("Update the following properties", item.props);
            });
          }
          promise = promise.then(() => {
            return this.onSetPropsCB(item.props);
          });
          if (this.enableLogging) {
            promise = promise.then(() => {
              console.log("Update finish!");
              console.groupEnd();
            });
          }
          break;
        case "reset_props":
          if (this.enableLogging) {
            promise = promise.then(() => {
              this.group(`Reset store properties`);
              console.log("Start");
            });
          }
          promise = promise.then(() => {
            return this.onResetPropsCB();
          });
          if (this.enableLogging) {
            promise = promise.then(() => {
              console.log("Finish!");
              console.groupEnd();
            });
          }
          break;
        case "exec_func":
          if (this.enableLogging) {
            promise = promise.then(() => {
              this.group(`Execute the function ${item.func}`);
              console.log(`Arguments:`, item.args);
            });
          }
          promise = promise.then(() => {
            return this.onExecFuncCB(item.func, item.args);
          });
          if (this.enableLogging) {
            promise = promise.then(() => {
              console.log("Finish!");
              console.groupEnd();
            });
          }
          break;
        default:
          console.error("Invalid message type! Get:", item.type);
      }
    }

    if (this.enableLogging) {
      promise = promise.then(() => {
        console.groupEnd();
      });
    }
    return promise;
  };

  /**
   * Handle response we get from the otherside for the request we made previously
   * @param version
   * @param payload
   */
  private handleResponse = (
    version: number,
    payload: { id: string; success: boolean; response: any }
  ) => {
    let requestId = payload.id;

    if (this.enableLogging) {
      this.group(
        `Receive a ${
        payload.success ? "success" : "failure"
        } response for request ${requestId}`
      );
      console.log("Response:", payload.response);
    }

    if (this.requests.has(requestId)) {
      let [requestVersion, resolve, reject] = this.requests.get(requestId)!;
      this.requests.delete(requestId);

      if (payload.success) {
        resolve({
          requestVersion,
          responseVersion: version,
          response: payload.response,
        });
      } else {
        reject({
          message: "error while handling the request",
          requestVersion,
          responseVersion: version,
          type: "request_error",
          response: payload.response,
        });
      }
    } else {
      console.error(
        `Ignore the response for the request ${requestId}, probably because of bugs`
      );
    }
  };

  private group = (msg: string) => {
    const now = new Date();
    const offsetMs = now.getTimezoneOffset() * 60 * 1000;
    const timeLocal = new Date(now.getTime() - offsetMs);
    console.groupCollapsed(
      `%c[${timeLocal.toISOString().slice(11, 23)}]`,
      "color: #52c41a",
      msg
    );
  };

  private logRoot = (...args: any[]) => {
    const now = new Date();
    const offsetMs = now.getTimezoneOffset() * 60 * 1000;
    const timeLocal = new Date(now.getTime() - offsetMs);
    console.log(
      `%c[${timeLocal.toISOString().slice(11, 23)}]`,
      "font-weight: 600; color: #52c41a",
      ...args
    );
  };
}
