import React from "react";
import { Input } from "antd";
import {
  SearchOutlined,
  LoadingOutlined,
  CloseCircleTwoTone,
} from "@ant-design/icons";

const defaultProps = {
  triggerSearchWindow: 200,
};
interface Props extends Readonly<typeof defaultProps> {
  size?: "middle" | "large" | "small";
  placeholder?: string;
  // default value of this input search
  value?: string;

  // change to edit mode on focus
  view2editOnFocus?: boolean;

  // trigger the search
  onSearch?: (query: string) => Promise<void>;
  // use to clear the search when users empty their query
  onClearSearch?: () => void;
  // callback when this input is focus
  onFocus?: () => void;
  // callback when this input lose focus
  onFocusOut?: () => void;
  // callback when user clear out the selection via a button in this input
  onClearSelection?: () => void;
}
interface State {
  query: string;
  mode: "edit" | "view";
  loading: boolean;
}

export class InputSearch extends React.Component<Props, State> {
  public static defaultProps = defaultProps;
  private delayFunc: NodeJS.Timeout | null = null;

  constructor(props: Props) {
    super(props);

    this.state = {
      // if now value, then it is in the view mode
      query: "",
      mode: props.value === undefined ? "edit" : "view",
      loading: false,
    };
  }

  /** Change this input to view mode */
  change2viewMode = () => {
    this.setState({
      mode: "view",
      query: ""
    });
  }

  onChange = (e: any) => {
    if (this.state.mode === "view" && this.props.value !== undefined) {
      // change the value when the query is not set, convert it to the edit/search mode
      if (e.target.value.length < this.props.value.length) {
        // user press backspace
        this.setState({ query: "", mode: "edit" });
      } else {
        this.setState({
          query: e.target.value.substring(this.props.value.length),
          mode: "edit"
        });
      }
    } else {
      this.setState({ query: e.target.value });
    }
    let query = e.target.value.trim();

    if (this.props.onSearch !== undefined && query.length > 0) {
      this.setState({ loading: true });
      this.search();
    }

    if (this.props.onClearSearch !== undefined && query.length === 0) {
      // we may have case where on clear search was issue first, then the promise search finish
      // and overwrite the result.
      if (this.delayFunc !== null) {
        clearTimeout(this.delayFunc);
      }

      this.props.onClearSearch();
    }
  };

  onFocus = () => {
    if (this.props.view2editOnFocus) {
      this.setState({ query: "" });
    }
    if (this.props.onFocus !== undefined) {
      this.props.onFocus();
    }
  }

  detectEscKey = (e: any) => {
    if (e.keyCode === 27 && this.props.value !== undefined) {
      // clear the query to back to the view mode, only if the view mode is available
      this.change2viewMode();
      e.preventDefault();
      e.stopPropagation();
      if (this.props.onFocusOut) {
        this.props.onFocusOut();
      }
    }
  };

  /**
   * Clear the value that is current selected. This is difference to the behaviour that
   * we clear the display when switch from the view mode to the edit mode (we not actually clear the value)
   * but hide it so user can search.
   */
  clearValue = () => {
    // change to edit mode
    this.setState({ mode: "edit", query: "" });
    if (this.props.onClearSelection !== undefined) {
      this.props.onClearSelection();
    }
  };

  search = () => {
    if (this.delayFunc !== null) {
      clearTimeout(this.delayFunc);
    }

    this.delayFunc = setTimeout(() => {
      this.props.onSearch!(this.state.query || "").then(() => {
        this.setState({ loading: false });
      });
    }, this.props.triggerSearchWindow);
  };

  render() {
    let isViewMode = this.state.mode === "view"
    let props = {
      placeholder: this.props.placeholder,
      size: this.props.size,
      value: isViewMode ? this.props.value : this.state.query,
      onChange: this.onChange,
      onFocus: this.onFocus,
      // onBlur: this.props.onFocusOut,
      style: {},
    };

    if (isViewMode) {
      props.style = { border: "1px solid #52c41a" };
    }

    let suffix = this.state.loading ? (
      <LoadingOutlined />
    ) : isViewMode && !this.props.view2editOnFocus ? (
      <CloseCircleTwoTone twoToneColor="#cf1322" onClick={this.clearValue} />
    ) : (
          <SearchOutlined />
        );
    return <Input {...props} suffix={suffix} onKeyDown={this.detectEscKey} />;
  }
}
