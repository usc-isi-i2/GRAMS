import orjson

from sm_annotator.base_app import BaseApp


class FoldableList(BaseApp):
    def __init__(self, dev: bool = False):
        super().__init__("foldable_list", dev=dev)

    def show_list(self, items: list, header: str=None):
        props = {"items": items}
        if header is not None:
            props['header'] = header
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "wait_for_client_ready",
            },
            {
                "type": "set_props",
                "props": props
            }
        ]).decode())