import { Socket, Store } from "../library";
import { action, toJS } from "mobx";

export type ListItem = string | List;

export interface List {
  header: string;
  items: ListItem[];
}

export interface StoreProps {
  header?: string;
  items: ListItem[];
}

export class AppStore extends Store<StoreProps> {
  constructor(socket: Socket, defaultProps: StoreProps) {
    super(socket, defaultProps, undefined, AppStore.deserialize);
  }

  // deserialize the data from the server
  public static deserialize(socket: Socket, prop: string, data: any) {
    return data;
  }
}