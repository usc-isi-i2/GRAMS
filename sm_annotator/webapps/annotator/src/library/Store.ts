import { action, observable } from "mobx";
import { Socket } from "./Socket";

export class Store<T = { [prop: string]: any }> {
  @observable version: number = 0;
  @observable props: T = {} as T;

  protected socket: Socket;
  private defaultProps: T;
  private onUpdateProps?: (store: Store<T>, updateProps: string[]) => void;
  private deserialize: (socket: Socket, name: string, data: any) => any;

  constructor(socket: Socket, defaultProps: T, onUpdateProps?: (store: Store<T>, updateProps: string[]) => void, deserialize?: (socket: Socket, name: string, data: any) => any) {
    this.defaultProps = defaultProps;
    this.socket = socket;

    this.onUpdateProps = onUpdateProps;
    this.deserialize = deserialize || ((_socket, _name, data) => data);

    for (let [prop, value] of Object.entries(defaultProps)) {
      (this.props as any)[prop] = this.deserialize(this.socket, prop, value);
    }

    if (this.onUpdateProps !== undefined && Object.keys(defaultProps).length > 0) {
      this.onUpdateProps(this, Object.keys(defaultProps));
    }
  }

  @action
  public setProps(props: { [prop: string]: any }, nextVersion?: number) {
    if (nextVersion !== undefined && this.version + 1 !== nextVersion) {
      throw new Error("Bug in the code. The next version is not an incremental number of the current version");
    }

    this.version += 1;
    for (let [prop, value] of Object.entries(props)) {
      (this.props as any)[prop] = this.deserialize(this.socket, prop, value);
    }

    if (this.onUpdateProps !== undefined) {
      this.onUpdateProps(this, Object.keys(props));
    }
  }

  @action
  public resetProps(nextVersion?: number) {
    if (nextVersion !== undefined && this.version + 1 !== nextVersion) {
      throw new Error("Bug in the code. The next version is not an incremental number of the current version");
    }

    this.version += 1;
    this.props = { ...this.defaultProps };
    if (this.onUpdateProps !== undefined) {
      this.onUpdateProps(this, Object.keys(this.props));
    }
  }
}
