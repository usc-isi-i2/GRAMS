export interface Tunnel {
  send_msg: (msg: string) => number;
  send_msg_with_version: (version: number, msg: string) => void;
  on_receive: (cb: (version: number, msg: string) => Promise<void>) => void;
}

export class RecordTunnel implements Tunnel {
  private tunnel: Tunnel;
  private records: any[];

  constructor(tunnel: Tunnel) {
    this.tunnel = tunnel;
    this.records = [];
  }

  send_msg = (msg: string) => {
    let version = this.tunnel.send_msg(msg);
    this.records.push({
      type: "send_msg",
      msg, version
    });
    return version;
  }

  send_msg_with_version = (version: number, msg: string) => {
    this.records.push({
      type: 'send_msg_with_version',
      version, msg
    });
    this.tunnel.send_msg_with_version(version, msg);
  }

  on_receive = (cb: (version: number, msg: string) => void) => {
    this.tunnel.on_receive((version, msg) => {
      this.records.push({
        type: "on_receive",
        version,
        msg,
      });
      cb(version, msg);
      return Promise.resolve();
    });
  }

  serialize = () => {
    return JSON.stringify(this.records);
  }
}

/**
 * A replay tunnel that:
 * 1. Replay the incoming message on top of the recording list as soon as there is a callback registered
 * 2. When a message is sent, it is looking for the next item in the recording list and replay that item
 *    if it is an incoming message.
 */
export class ReplayTunnel implements Tunnel {
  private records: any[];
  private onReceiveCB?: (version: number, msg: string) => Promise<void>;

  constructor(records: any[]) {
    this.records = records;
  }

  send_msg = (msg: string) => {
    if (this.onReceiveCB === undefined) {
      alert("The receive callback must be set before sending message");
      return;
    }

    let record = this.records.shift();
    let iscorrect = record.type === 'send_msg' && record.msg === msg;
    if (!iscorrect) {
      alert("Invalid message order!");
      console.log('record=', record, 'msg=', msg);
      throw new Error("Invalid message order!");
    }

    if (this.records.length > 0 && this.records[0].type === 'on_receive') {
      setTimeout(() => {
        let promise = Promise.resolve();
        while (this.records.length > 0 && this.records[0].type === 'on_receive') {
          let record = this.records.shift();
          promise = promise.then(() => {
            return this.onReceiveCB!(record.version, record.msg);
          });
        }
      }, 50);
    }

    return record.version;
  }

  send_msg_with_version = (version: number, msg: string) => {
    if (this.onReceiveCB === undefined) {
      alert("The receive callback must be set before sending message");
      return;
    }

    if (this.onReceiveCB === undefined) {
      alert("The receive callback must be set before sending message");
      return;
    }

    let record = this.records.shift();
    let iscorrect = record.type === 'send_msg' && record.msg === msg;
    if (!iscorrect) {
      alert("Invalid message order!");
      console.log('record=', record, 'msg=', msg);
      throw new Error("Invalid message order!");
    }

    let promise = Promise.resolve();
    while (this.records.length > 0 && this.records[0].type === 'on_receive') {
      let record = this.records.shift();
      promise = promise.then(() => {
        return this.onReceiveCB!(record.version, record.msg);
      });
    }
  }

  on_receive = (cb: (version: number, msg: string) => Promise<void>) => {
    let promise = Promise.resolve();
    this.onReceiveCB = cb;
    // fire the first record if it is the message order
    while (this.records.length > 0 && this.records[0].type === 'on_receive') {
      let record = this.records.shift();
      promise = promise.then(() => {
        return this.onReceiveCB!(record.version, record.msg);
      });
    }
  }
}