import { LinkOutlined } from "@ant-design/icons";
import { List } from "antd";
import React from "react";
import { THEME } from "../../env";
import { ClickOutsideDetector } from "./ClickOutsideDetector";
import { InputSearch } from "./InputSearch";
import { ExternalLink } from "./ExternalLink";

export interface Record {
  id: string;
  label: string;
  url?: string;
  description?: string;
  style?: object;
}

interface Props {
  record?: Record;
  onSelectRecord: (record: Record) => void;
  findByName: (query: string) => Promise<Record[]>;
}

interface State {
  records?: Record[];
  showResult: boolean;
}

class SequentialFuncInvoker {
  private timer: number = 0;
  public exec<T>(fn: () => Promise<T>): Promise<T | undefined> {
    this.timer += 1;
    let calledAt = this.timer;

    return fn().then((result: any) => {
      if (calledAt < this.timer) {
        return undefined;
      }

      return result;
    });
  }
}

export class RecordSearch extends React.Component<Props, State> {
  private inputSearchRef = React.createRef<InputSearch>();
  public state: State = { showResult: false };
  public seqInvoker = new SequentialFuncInvoker();
  public styles = {
    container: {
      boxSizing: "border-box" as "border-box",
      marginTop: 4,
      width: "100%",
      backgroundColor: "#fff",
      zIndex: 1050,
      position: "absolute" as "absolute",
      borderRadius: 4,
      boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
    },
    searchItem: {
      maxWidth: "90%",
    },
  };

  search = (query: string): Promise<void> => {
    return this.seqInvoker
      .exec<Record[]>(() => {
        return this.props.findByName(query);
      })
      .then((records) => {
        this.setState({ records, showResult: true });
      });
  };

  clearSearch = () => {
    this.setState({ records: undefined, showResult: false });
  };

  showSearchResult = () => {
    this.setState({ showResult: true });
  };

  hideSearchResult = () => {
    this.setState({ showResult: false });
  };

  selectRecord = (record: Record) => {
    return (e: any) => {
      if (this.props.record !== undefined && this.props.record.id === record.id) {
        // select the same record; the react state doesn't change, so the InputSearch doesn't updated from the edit mode to the view mode
        // so we need to manually tell change to the view mode
        // and don't need to fire the onSelectRecord event
        this.inputSearchRef.current!.change2viewMode();
      } else {
        this.props.onSelectRecord(record);
      }
      this.hideSearchResult();
    };
  };

  render() {
    let display =
      (this.props.record !== undefined || this.state.records !== undefined) &&
        this.state.showResult
        ? "inherit"
        : "none";
    let searchResult = (
      <div style={{ ...this.styles.container, display }}>
        <List
          size="small"
          bordered={true}
          dataSource={(this.props.record !== undefined
            ? [this.props.record]
            : []
          ).concat(this.state.records || [])}
          style={THEME === 'dark' ? { background: 'black' } : {}}
          renderItem={(record, index) => {
            let style =
              index === 0 && this.props.record !== undefined
                ? { borderBottom: "3px double" }
                : {};
            let link = null;
            if (record.url !== undefined) {
              link = <ExternalLink url={record.url}>
                <LinkOutlined />
              </ExternalLink>
            }
            return (
              <List.Item
                onClick={this.selectRecord(record)}
                style={{ cursor: "pointer", ...style, ...record.style }}
              >
                <List.Item.Meta
                  style={this.styles.searchItem}
                  title={
                    <React.Fragment>
                      {record.label}&nbsp;&nbsp;
                      {link}
                    </React.Fragment>
                  }
                  description={record.description}
                />
              </List.Item>
            );
          }}
        />
      </div>
    );

    return (
      <div style={{ position: "relative" }}>
        <ClickOutsideDetector onFocusOut={this.hideSearchResult}>
          <div>
            <InputSearch
              ref={this.inputSearchRef}
              size="middle"
              placeholder={`search`}
              onSearch={this.search}
              onClearSearch={this.clearSearch}
              onFocus={this.showSearchResult}
              onFocusOut={this.hideSearchResult}
              view2editOnFocus={true}
              value={
                this.props.record === undefined
                  ? undefined
                  : this.props.record.label
              }
            />
          </div>
          {searchResult}
        </ClickOutsideDetector>
      </div>
    );
  }
}