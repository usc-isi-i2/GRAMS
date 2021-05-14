import React from "react";


export class ClickOutsideDetector extends React.Component<{ onFocusOut: () => void }> {
  wrapperRef: any;

  constructor(props: any) {
    super(props);

    this.wrapperRef = React.createRef();
    this.handleClickOutside = this.handleClickOutside.bind(this);
  }

  componentDidMount() {
    document.addEventListener('mousedown', this.handleClickOutside);
  }

  componentWillUnmount() {
    document.removeEventListener('mousedown', this.handleClickOutside);
  }

  /**
   * Alert if clicked on outside of element
   */
  handleClickOutside(event: any) {
    // if running inside a shadow dom, event.target is retargetted to return root of the shadow dom.
    // so we need to get it fromo the event.path
    let target = event.path[0];
    if (this.wrapperRef && !this.wrapperRef.current.contains(target)) {
      this.props.onFocusOut();
    }
  }

  render() {
    return <div ref={this.wrapperRef}>{this.props.children}</div>;
  }
}