import React from "react";

interface Props {
  url: string;
  style?: React.CSSProperties;
  action?: React.ReactNode;
}

export class ExternalLink extends React.Component<Props, {}> {
  render() {
    let altLink = null;
    if (this.props.url.startsWith("http://www.wikidata.org/")) {
      let altUrl = this.props.url.split("/");
      altLink = <React.Fragment>
        &nbsp;&nbsp;
        <a href={`https://ringgaard.com/kb/${altUrl[altUrl.length - 1]}`} target='_blank' rel="noopener noreferrer" style={this.props.style}>▒</a>
        &nbsp;&nbsp;
      </React.Fragment>
    }

    return <React.Fragment>
      <a href={this.props.url} target='_blank' rel="noopener noreferrer" style={this.props.style}>
        {this.props.children}
      </a>
      {altLink}
      {this.props.action}
    </React.Fragment>
  }
}