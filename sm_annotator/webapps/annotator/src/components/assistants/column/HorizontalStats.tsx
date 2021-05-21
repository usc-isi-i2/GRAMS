import React from "react";

interface Props {
  stats: { [k: string]: string },
  style?: React.CSSProperties
}

export class HorizontalStats extends React.Component<Props, {}> {
  render() {
    let keys = [];
    let values = [];

    for (let [k, v] of Object.entries(this.props.stats)) {
      keys.push(<th key={k}>{k}</th>);
      values.push(<td key={k}>{v}</td>);
    }

    return <table className="lightweight-table" style={this.props.style}>
      <thead>
        <tr>{keys}</tr>
      </thead>
      <tbody>
        <tr>{values}</tr>
      </tbody>
    </table>
  }
}