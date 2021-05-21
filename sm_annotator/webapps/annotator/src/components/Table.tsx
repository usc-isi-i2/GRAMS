import { Modal, Popover, Tag } from "antd";
import Table, { TablePaginationConfig } from "antd/lib/table";

import memoizeOne from "memoize-one";
import { inject, observer } from "mobx-react";
import React from "react";
import { Store } from "../library";
import {
  StoreProps,
  Table as StoreTableProps,
  TableCell,
  TableColumn,
  TableRow,
  AppStore,
  EntityStore,
  TablePagination,
  TableFilter,
  Entity
} from "../models";
import { ExternalLink } from "./primitives/ExternalLink";
import ColumnAssistant from "./assistants/column/ColumnAssistant";
import { EntityComponentExternalOpener, EntityComponent } from "./Entity";
import { getIndentIndicator } from "./assistants/column/ColumnTypeTree";
import { toJS } from "mobx";
import { timingSafeEqual } from "crypto";

export interface Props {
  table?: StoreTableProps;
  store?: AppStore;
}

interface ModalColumnInfo {
  columnId: number;
}

interface ModalEntityInfo {
  entity: Entity;
}

interface State {
  modal: { visible: boolean; info: ModalColumnInfo | ModalEntityInfo | null };
}

@inject((provider: { store: AppStore }) => ({
  store: provider.store,
  table: provider.store.props.table,
}))
@observer
export default class TableComponent extends React.Component<Props, State> {
  public static defaultProps = {
    columns: [],
  };
  private container = React.createRef<HTMLDivElement>();

  constructor(props: Props) {
    super(props);
    this.state = {
      modal: { visible: false, info: null },
    };
  }

  get table() {
    return this.props.table!;
  }

  get shouldOpenEntityNewPage() {
    // change to true will open it in new page, false to open in a modal
    return false;
  }

  componentDidMount() {
    // this.updateHeadersPositions();
  }

  componentDidUpdate(prevProps: Props, prevState: State) {
    // TODO: should check if the content of the table has changed
    // this.updateHeadersPositions();
  }

  getPagination = (table: StoreTableProps) => {
    return {
      total: table.pagination.total,
      current: table.pagination.current,
      pageSize: table.pagination.pageSize,
      pageSizeOptions: ["5", "10", "20", "50", "100", '200', '500', '1000'],
      showSizeChanger: true,
      showTotal: (total: number) => `Total ${total} items`,
    };
  };

  updateHeadersPositions = () => {
    if (this.container.current === null) {
      return;
    }

    let offset = this.container.current.getBoundingClientRect().left;
    let dataNodePositions = Array.from(
      this.container.current.querySelectorAll("table thead th")
    ).map((el) => {
      let bbox = el.getBoundingClientRect();
      return {
        left: bbox.left - offset,
        centerLeft: bbox.left + bbox.width / 2 - offset,
      };
    });

    // remove the first column
    dataNodePositions.shift();
    let store = this.props.store!;
    let prevDataNodePositions = store.currentGraph.dataNodePositions;

    if (
      prevDataNodePositions !== undefined &&
      prevDataNodePositions.length === dataNodePositions.length
    ) {
      let isEqual = true;
      for (let i = 0; i < prevDataNodePositions.length; i++) {
        let v0 = prevDataNodePositions[i].centerLeft;
        let v1 = dataNodePositions[i].centerLeft;

        if (v0 !== v1) {
          isEqual = false;
          break;
        }
      }

      if (isEqual) {
        return;
      }
    }

    this.props.store!.setProps({
      graph: { ...store.currentGraph, dataNodePositions },
    });
  };

  /** Tell the table to fetch the data */
  changePage = (
    pagination: TablePaginationConfig,
    _filters?: any,
    _sorter?: any
  ) => {
    this.table.updatePagination({
      current: pagination.current!,
      pageSize: pagination.pageSize!,
    });
  };

  getColumns = memoizeOne((tableId: string, columns: TableColumn[]) => {
    return columns.map((col, colindex) => {
      // minus 1 because we introduce a fake column
      let title = (
        <span
          onClick={
            colindex >= 1 ? this.modalViewColumn(colindex - 1) : undefined
          }
        >
          {col.title}
        </span>
      );

      return {
        title: title,
        dataIndex: col.dataIndex,
        render: (cell: number | string | TableCell, record: TableRow) => {
          if (typeof cell !== "object") {
            return <Tag color="geekblue">{cell}</Tag>;
          }

          let components = cell.links.flatMap((link, index) => {
            let prefix =
              index === 0
                ? cell.value.substring(0, link.start)
                : cell.value.substring(cell.links[index - 1].end, link.start);
            let linksurface = cell.value.substring(link.start, link.end);
            let infix = (
              <a
                key={index}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                dangerouslySetInnerHTML={{
                  __html: linksurface.trim() === "" ? "&blank;" : linksurface,
                }}
                style={link.entity === null ? { color: "#aaa" } : {}}
              />
            );
            return [prefix, infix];
          });

          if (cell.links.length === 0) {
            components.push(cell.value);
          } else {
            components.push(
              cell.value.substring(cell.links[cell.links.length - 1].end)
            );
          }

          let content = (
            <div className="popover-table">
              <table className="lightweight-table">
                <thead>
                  <tr>
                    <th>KG Entity</th>
                    <th>Class</th>
                  </tr>
                </thead>
                <tbody>
                  {cell.links
                    .filter((link) => link.entity !== null)
                    .map((link) => {
                      let entity = cell.metadata.entities[link.entity!];
                      let action = <EntityComponentExternalOpener
                        entityURI={entity.uri} open={this.shouldOpenEntityNewPage ? undefined : this.modalViewEntity}
                        autoHighlight={true} />
                      return (
                        <tr key={entity.uri}>
                          <td>
                            <ExternalLink url={entity.uri} action={action}>
                              {entity.label}
                            </ExternalLink>
                          </td>
                          <td>
                            <ul>
                              {entity.types.map((type) => {
                                return (
                                  <li key={type.uri}>
                                    <ExternalLink url={type.uri}>
                                      {getIndentIndicator(type.depth)} {type.label}
                                    </ExternalLink>
                                  </li>
                                );
                              })}
                            </ul>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          );

          return (
            <div>
              <Popover
                content={content}
                getPopupContainer={(trigger) => this.container.current!}
              >
                <span>{components}</span>
              </Popover>
            </div>
          );
        },
      };
    });
  });

  modalViewColumn = (columnId: number) => {
    return () => {
      this.setState({ modal: { visible: true, info: { columnId: columnId } } });
    };
  };

  modalViewEntity = (entity: Entity) => {
    this.setState({ modal: { visible: true, info: { entity } } });
  }

  modalHide = () => {
    this.setState({
      modal: { visible: false, info: null },
    });
  };

  render() {
    let table = this.props.table!;
    let metadata = table.metadata;
    let wikidataInfo = null;

    if (metadata.entity !== null) {
      let action = <EntityComponentExternalOpener
        key={metadata.entity.uri}
        entityURI={metadata.entity.uri} open={this.shouldOpenEntityNewPage ? undefined : this.modalViewEntity}
        autoHighlight={true} />

      wikidataInfo = (
        <span>
          Wikidata page:{" "}
          <ExternalLink url={metadata.entity.uri} action={action}>{metadata.entity.label}</ExternalLink>
        </span>
      );
    }

    let modal = null;
    if (this.state.modal.info !== null) {
      let props = {
        getContainer: () => this.container.current!,
        visible: this.state.modal.visible,
        cancelButtonProps: { style: { display: "none " } },
        onOk: this.modalHide,
        onCancel: this.modalHide,
        width: "90%",
        style: { top: 8 },
        maskClosable: true,
      }

      if ('columnId' in this.state.modal.info) {
        // showing column
        modal = <Modal
          title={table.getColumnById(this.state.modal.info.columnId).title}
          {...props}
        >
          <ColumnAssistant columnId={this.state.modal.info.columnId} />
        </Modal>
      } else {
        modal = <Modal
          title={`Entity`}
          {...props}>
          <EntityComponent entity={this.state.modal.info.entity} highlightProps={this.props.store!.relevantProps} />
        </Modal>
      }
    }

    return (
      <div ref={this.container}>
        <Table
          size="small"
          dataSource={table.rows}
          rowKey={table.rowKey}
          pagination={this.getPagination(table)}
          columns={this.getColumns(table.id, table.columns)}
          loading={table.loading}
          onChange={this.changePage}
        />
        <p style={{ textAlign: "center" }}>
          Wikipedia page:{" "}
          <a href={metadata.url} target="_blank" rel="noopener noreferrer">
            {metadata.title}
          </a>
          . {wikidataInfo}
        </p>
        {modal}
      </div>
    );
  }
}
