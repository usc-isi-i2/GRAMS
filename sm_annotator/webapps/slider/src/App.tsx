import React from "react";
import { inject, observer } from "mobx-react";
import { Socket } from "./library";
import AppCSS from "./App.css";
import { AppStore } from "./models";

interface Props {
  socket: Socket;
  store: AppStore;
  min: number;
  max: number;
  index: number;
  description: string;
}

interface State {
  index: string;
}

@inject((provider: { store: AppStore }) => ({
  min: provider.store.props.min,
  max: provider.store.props.max,
  index: provider.store.props.index,
  description: provider.store.props.description,
}))
@observer
export default class App extends React.Component<Props, State> {
  public state: State = {
    index: ''
  };

  next = () => {
    this.props.store.setIndex(this.props.index + 1);
  }

  prev = () => {
    this.props.store.setIndex(this.props.index - 1);
  }

  updateIndex = (e: any) => {
    this.setState({
      index: e.target.value.replace(/\D/g, '')
    });
  }

  navigate2index = (e: any) => {
    if (e.key === 'Enter' && this.state.index.length > 0) {
      let index = parseInt(this.state.index);
      // reset it
      this.setState({ index: '' });
      this.props.store.setIndex(index);
    }
  }

  handleKeyPress = (e: any) => {
    if (e.altKey) {
      if (e.keyCode === 190) {
        // press next
        this.next();
      }
      if (e.keyCode === 188) {
        // press previous
        this.prev();
      }
    }
  }

  componentDidMount() {
    document.addEventListener("keydown", this.handleKeyPress, false);
  }

  componentWillUnmount() {
    document.removeEventListener("keydown", this.handleKeyPress, false);
  }

  render() {
    return <div>
      <style type="text/css">{AppCSS}</style>
      <button className="btn btn-default  " onClick={this.prev}>Previous (alt + &#60;)</button>
      <button className="btn btn-default ml-2" onClick={this.next}>Next (alt + &#62;)</button>
      <span className="ml-2">Index: {this.props.index} ([{this.props.min}, {this.props.max}])</span>
      <input className="ml-2 form-control" value={this.state.index} onChange={this.updateIndex} onKeyPress={this.navigate2index} />
      <span className="ml-2" dangerouslySetInnerHTML={{ __html: this.props.description }} />
    </div>;
  }
}
