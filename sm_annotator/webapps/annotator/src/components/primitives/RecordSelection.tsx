import { List } from "antd";
import React from "react";
import { THEME } from "../../env";
import { ClickOutsideDetector } from "./ClickOutsideDetector";
import { InputSearch } from "./InputSearch";
import memoizeOne from "memoize-one";
import Fuse from "fuse.js";

export interface Record {
  id: string;
  label: string;
  style?: object;
}

interface Props {
  selectRecord?: Record;
  records: Record[];
  onSelectRecord: (record: Record) => void;
  rankByName: (query: string) => number[];
  // style of the record selection box
  style?: object;
  renderRecordTitle?: (record: Record) => React.Component;
}

interface State {
  recordOrders?: number[];
  showResult: boolean;
}

export const defaultRankByName = memoizeOne((records: Record[]) => {
  let fuse = new Fuse(records, {
    includeScore: true,
    keys: ['label']
  });

  return (query: string) => {
    return fuse.search(query).map(x => x.refIndex);
  }
});

export class RecordSelection extends React.Component<Props, State> {
  private inputRef = React.createRef<InputSearch>();
  public state: State = { showResult: false };
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

  search = (query: string) => {
    this.setState({ recordOrders: this.props.rankByName(query), showResult: true });
    return Promise.resolve();
  };

  showSearchResult = () => {
    this.setState({ showResult: true });
  };

  hideSearchResult = () => {
    this.setState({ showResult: false });
  };

  selectRecord = (record: Record) => {
    return (e: any) => {
      this.props.onSelectRecord(record);
      this.inputRef.current!.change2viewMode();
      this.hideSearchResult();
    };
  };

  getRecordsComponent = () => {
    let display = this.state.showResult ? "inherit" : "none";
    let records = this.state.recordOrders === undefined ? this.props.records : this.state.recordOrders.map((index) => this.props.records[index]);
    let renderRecordTitle = this.props.renderRecordTitle === undefined ? (record: Record) => record.label : this.props.renderRecordTitle;

    return <div style={{ ...this.styles.container, display }}>
      <List
        size="small"
        bordered={true}
        dataSource={records}
        style={THEME === 'dark' ? { background: 'black' } : {}}
        renderItem={(record, index) => {
          return (
            <List.Item
              onClick={this.selectRecord(record)}
              style={{ cursor: "pointer", ...record.style }}
            >
              <List.Item.Meta
                style={this.styles.searchItem}
                title={renderRecordTitle(record)}
                description={null}
              />
            </List.Item>
          );
        }}
      />
    </div>
  }

  render() {
    return (
      <div style={{ position: "relative", ...this.props.style }}>
        <ClickOutsideDetector onFocusOut={this.hideSearchResult}>
          <div>
            <InputSearch
              ref={this.inputRef}
              size="middle"
              placeholder={`search`}
              onSearch={this.search}
              onFocus={this.showSearchResult}
              onFocusOut={this.hideSearchResult}
              view2editOnFocus={true}
              value={
                this.props.selectRecord === undefined
                  ? undefined
                  : this.props.selectRecord.label
              }
            />
          </div>
          {this.getRecordsComponent()}
        </ClickOutsideDetector>
      </div>
    );
  }
}