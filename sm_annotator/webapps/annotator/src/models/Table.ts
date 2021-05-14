import { flow, observable } from "mobx";
import { CancellablePromise } from "mobx/lib/api/flow";
import { Socket } from "../library";
import { AppStore } from "./Store";
import { Resource, ExcerptEntity } from "./Entity";
import memoizeOne from "memoize-one";

export interface TableCell {
  value: string;
  links: { start: number; end: number; href: string; entity: string | null }[];
  metadata: {
    entities: { [uri: string]: ExcerptEntity }
  }
}

export interface TableRow {
  rowId: number;
  data: TableCell[];
}

export interface TableColumn {
  title: string;
  columnId: number;
  dataIndex: string | string[];
}

export interface TableColumnTypeFilter {
  [columnId: number]: {
    [cls: string]: "include" | "exclude"
  };
}

interface TableColumnRelationFilterData {
  [columnId: number]: {
    // key is tuple (endpoint (* when you want to apply to all), pred0, pred1)
    incoming: { [key: string]: "include" | "exclude" },
    outgoing: { [key: string]: "include" | "exclude" },
  }
}

export class TableColumnRelationFilter {
  static WILDCARD_ENDPOINT = "*";

  @observable data: TableColumnRelationFilterData;

  constructor(data: TableColumnRelationFilterData) {
    this.data = data;
  }

  shallowClone() {
    return new TableColumnRelationFilter(Object.assign({}, this.data));
  }

  getFilterOp(columnId: number, direction: "incoming" | "outgoing", endpoint: number | string, predicate0: string, predicate1: string): "include" | "exclude" | undefined {
    if (this.data[columnId] === undefined) {
      return undefined;
    }

    let xfilter = this.data[columnId][direction];
    let key = JSON.stringify([endpoint, predicate0, predicate1]);
    return xfilter[key];
  }

  setFilterOp(columnId: number, direction: "incoming" | "outgoing", endpoint: number | string, predicate0: string, predicate1: string, op: "include" | "exclude") {
    if (this.data[columnId] === undefined) {
      this.data[columnId] = { incoming: {}, outgoing: {} }
    }

    let xfilter = this.data[columnId][direction];
    let key = JSON.stringify([endpoint, predicate0, predicate1]);
    xfilter[key] = op;
  }

  removeFilterOp(columnId: number, direction: "incoming" | "outgoing", endpoint: number | string, predicate0: string, predicate1: string) {
    if (this.data[columnId] === undefined) {
      return;
    }

    let xfilter = this.data[columnId][direction];
    let key = JSON.stringify([endpoint, predicate0, predicate1]);
    delete xfilter[key];
  }

  serialize() {
    // serialize the data to send to server
    return Object.entries(this.data)
      .flatMap((args) => {
        let columnId = parseInt(args[0]);
        return Object.entries(args[1])
          .flatMap((args2) => {
            let direction = args2[0];
            return Object.entries(args2[1] as { [key: string]: "include" | "exclude" })
              .flatMap((args3) => {
                let [endpoint, predicate0, predicate1] = JSON.parse(args3[0]);
                let op = args3[1];

                return {
                  columnId,
                  endpoint,
                  pred1: predicate0,
                  pred2: predicate1,
                  direction,
                  op,
                }
              });
          });
      })
  }
}

export interface TableColumnLinkFilter {
  [columnId: number]: "noLink" | "hasLink" | "noEntity" | "hasEntity"
}

export interface TableFilter {
  columnRelationFilters: TableColumnRelationFilter;
  columnTypeFilters: TableColumnTypeFilter;
  columnLinkFilters: TableColumnLinkFilter;
}

interface InputPagination {
  // the no of the current page, start from 1
  current: number;
  pageSize: number;
}

export interface TablePagination extends InputPagination {
  // total number of records
  total: number;
}

export class Table {
  private socket: Socket;

  @observable id: string;
  @observable columns: TableColumn[];
  @observable rowKey: string;
  @observable totalRecords: number;
  @observable metadata: {
    title: string;
    url: string;
    entity: Resource | null;
  };
  @observable filters: TableFilter;
  @observable pagination: TablePagination;
  @observable loading: boolean;
  @observable rows: TableRow[];

  constructor(socket: Socket, props: any) {
    this.socket = socket;
    this.id = props.id;
    this.columns = props.columns;
    this.rowKey = props.rowKey;
    this.totalRecords = props.totalRecords;
    this.metadata = props.metadata;
    this.filters = {
      columnTypeFilters: {},
      columnRelationFilters: new TableColumnRelationFilter({}),
      columnLinkFilters: {}
    };
    this.pagination = {
      current: 1,
      pageSize: 5,
      total: props.totalRecords,
    };
    this.loading = false;
    this.rows = [];
  }

  public getColumnById = (columnId: number) => {
    // +1 as column id is actually column index shifted by 1 (they introduce a index column)
    return this.columns[columnId + 1];
  };

  public updatePagination(pagination: InputPagination) {
    return this.fetchData(pagination, undefined);
  }

  public updateFilter(filter: TableFilter) {
    return this.fetchData(
      { current: 1, pageSize: this.pagination.pageSize },
      filter
    );
  }

  public hasFilters = () => {
    return Object.keys(this.filters.columnTypeFilters).length > 0 ||
      Object.keys(this.filters.columnRelationFilters.data).length > 0 ||
      Object.keys(this.filters.columnLinkFilters).length > 0;
  }

  private fetchData: (
    pagination?: InputPagination,
    filters?: TableFilter
  ) => CancellablePromise<void> = flow(function* (
    this: Table,
    pagination?: InputPagination,
    filters?: TableFilter
  ) {
    this.loading = true;
    if (filters !== undefined) {
      this.filters = filters;
    }
    if (pagination !== undefined) {
      this.pagination.current = pagination.current;
      this.pagination.pageSize = pagination.pageSize;
    }

    let typeFilterArgs: any = [];
    let relFilterArgs: any = [];

    if (this.hasFilters()) {
      typeFilterArgs = Object.entries(this.filters.columnTypeFilters).flatMap(
        (args) => {
          return Object.entries(args[1]).map((map) => ({
            columnId: parseInt(args[0]),
            uri: map[0],
            op: map[1],
          }));
        }
      );

      relFilterArgs = this.filters.columnRelationFilters.serialize();
    }

    let resp = yield this.socket.request("/table", {
      offset: (this.pagination.current! - 1) * this.pagination.pageSize!,
      limit: this.pagination.pageSize,
      typeFilters: typeFilterArgs,
      relFilters: relFilterArgs,
      linkFilters: Object.assign({}, this.filters.columnLinkFilters)
    });

    this.rows = resp.response.rows;
    this.pagination.total = resp.response.total;
    this.loading = false;
  });
}
