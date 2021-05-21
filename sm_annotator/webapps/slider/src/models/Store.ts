import { Socket, Store } from "../library";
import { action, toJS } from "mobx";

export interface StoreProps {
  min: number;
  max: number;
  index: number;
  description: string;
}

export class AppStore extends Store<StoreProps> {
  constructor(socket: Socket, defaultProps: StoreProps) {
    super(socket, defaultProps, undefined, AppStore.deserialize);
  }

  @action
  public setIndex = (index: number) => {
    if (index >= this.props.min && index <= this.props.max) {
      this.props.index = index;
      this.socket.request("/view", { index });
    }
  }

  // deserialize the data from the server
  public static deserialize(socket: Socket, prop: string, data: any) {
    return data;
  }
}