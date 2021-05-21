import React from "react";
import { inject, observer } from "mobx-react";
import { Socket } from "./library";
import AppCSS from "./App.css";
import { AppStore, List, ListItem } from "./models";

export class Bullet extends React.Component<{ depth: number, type: "list-opened" | "list-closed" | "item" | "none", onClick?: () => void }, {}> {
  render() {
    let style: any = {
      marginLeft: 20 * this.props.depth,
    }
    let content = "";
    switch (this.props.type) {
      case "list-closed":
        content = "▼";
        style.transform = 'rotate(270deg)';
        break
      case "list-opened":
        content = "▼";
        break;
      case "item":
        content = "·";
        style.transform = 'scale(2.5) translate(0px, -1px)';
        break;
      case "none":
        content = "";
        break
    }
    return <span className={`list-item-bullet`} style={style} onClick={this.props.onClick}>
      {content}
    </span>
  }
}

@observer
export class ListComponent extends React.Component<{ items: ListItem[], depth: number }, { displayItems: { [x: number]: boolean } }> {
  public state: { displayItems: { [x: number]: boolean } } = {
    displayItems: {}
  };

  toggleDisplay = (index: number) => {
    this.setState({
      displayItems: {
        ...this.state.displayItems,
        [index]: !this.isDisplayed(index)
      }
    });
  }

  isDisplayed = (itemIndex: number) => {
    return !(this.state.displayItems[itemIndex] === false);
  }

  render() {
    if (this.props.items.length === 0) {
      return <div className="list-item">
        <Bullet depth={this.props.depth} type="none" />
        <code style={{ color: '#ad2102' }}>[empty list]</code>
      </div>;
    }

    let rows = this.props.items.map((item, index) => {
      if (typeof item === 'string') {
        return <div key={index} className="list-item">
          <Bullet depth={this.props.depth} type="item" />
          <div style={{ display: "inline-block" }} dangerouslySetInnerHTML={{ __html: item }}></div>
        </div>;
      }

      let subitems = null;
      if (!(this.state.displayItems[index] === false)) {
        subitems = <ListComponent items={item.items} depth={this.props.depth + 1} />;
      }
      return <React.Fragment key={index}>
        <div className="list-item">
          <Bullet depth={this.props.depth} type={this.isDisplayed(index) ? "list-opened" : "list-closed"} onClick={() => this.toggleDisplay(index)} />
          <div style={{ display: "inline-block" }} dangerouslySetInnerHTML={{ __html: item.header }}></div>
        </div>
        {subitems}
      </React.Fragment>
    });
    return <React.Fragment>{rows}</React.Fragment>
  }
}

interface AppProps {
  socket: Socket;
  store: AppStore;
  header?: string;
  items: ListItem[]
}

interface AppState {
}

@inject((provider: { store: AppStore }) => ({
  header: provider.store.props.header,
  items: provider.store.props.items
}))
@observer
export default class App extends React.Component<AppProps, AppState> {
  render() {
    return <div>
      <style type="text/css">{AppCSS}</style>
      {this.props.header !== undefined ? <div className="list-item">
        <div style={{ display: "inline-block" }} dangerouslySetInnerHTML={{ __html: this.props.header }}></div>
      </div> : null}
      <ListComponent items={this.props.items} depth={0} />
    </div>;
  }
}
