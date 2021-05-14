import AppCSS from "./App.css";

// uncomment for different themes
// import AntDCSS from "antd/dist/antd.dark.min.css";
// export const THEME: "dark" | "light" = "dark";

import AntDCSS from "antd/dist/antd.min.css";
export const THEME: "dark" | "light" = "light";

export const THEME_CSS = AntDCSS + "\n" + AppCSS;
