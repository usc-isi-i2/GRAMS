import os
from uuid import uuid4

import ipywidgets
import orjson
import ujson
from IPython.core.display import Javascript, display
from ipycallback import SlowTunnelWidget
from ipywidgets import HTML
from typing import List, Optional, TypedDict, Any, Callable
from grams.misc import deserialize_text
from sm_annotator.base_app import BaseApp


class _SliderApp(BaseApp):
    pass


class SliderApp(_SliderApp):
    def __init__(self, app: BaseApp, app_render_fn: str, dev: bool=False):
        self.output22 = ipywidgets.Output()
        super().__init__("slider", dev=dev)
        self.app = app
        self.app_render_fn = getattr(app, app_render_fn) if isinstance(app_render_fn, str) else app_render_fn
        self.index = 0
    
    def render(self, same_tab: bool = True, new_window: bool = False, shadow_dom: Optional[bool] = None):
        display(self.output22)
        if shadow_dom is None:
            shadow_dom = True if same_tab else False
        if not same_tab:
            # let the app decide if they want to set shadow_dom cause we are rendering it in new tab
            self.app.render(same_tab, new_window)
            # now we inject the parents
            display(self.tunnel, Javascript(f"""
            {self.RepeatUntilSuccess}
            function setupSliderApp() {{
                if (window.IPyApps === undefined || window.IPyApps.get('{self.app.app_id}') === undefined) {{
                    return false;
                }}
                
                let appwin = window.IPyApps.get('{self.app.app_id}');
                let container = appwin.document.getElementById('{self.app.app_id}');
                
                if (container === null) {{
                    return false;
                }}

                let div = appwin.document.createElement("div");
                div.id = '{self.app_id}';
                div.style = "margin-bottom: 8px";
                container.parentElement.prepend(div);
                
                let tunnel = window.IPyCallback.get('{self.tunnel.tunnel_id}');
                // use the tunnel first to send out the code, after the application is rendered, the listening function 
                // is going to be replaced by the listener in the application, so we don't have to worry.
                tunnel.on_receive(function (version, msg) {{
                    let payload = JSON.parse(msg);
                    if (payload.type !== 'set_source_code') {{
                        alert('invalid calling order. you need to set the source code first');
                        return;
                    }}
                    appwin.eval(payload.code);
                    appwin.{self.app_js_render_fn}('{self.app_id}', tunnel);
                }});
                return true;
            }}
            
            repeatUntilSuccess(setupSliderApp, 50, 10);
            """))
        else:
            self.app.render(same_tab, new_window, shadow_dom)
            display(self.tunnel, Javascript(f"""
            {self.RepeatUntilSuccess}
            
            function setupSliderApp() {{
                let container = window.document.getElementById('{self.app.app_id}');
                if (container === null || window.IPyCallback === undefined) {{
                    return false;
                }}
                
                let tunnel = window.IPyCallback.get('{self.tunnel.tunnel_id}');
                if (tunnel === undefined) {{
                    return false;
                }}
                
                let div = window.document.createElement("div");
                div.id = '{self.app_id}';
                div.style = "margin-bottom: 8px";
                container.parentElement.prepend(div);
                
                // use the tunnel first to send out the code, after the application is rendered, the listening function 
                // is going to be replaced by the listener in the application, so we don't have to worry.
                tunnel.on_receive(function (version, msg) {{
                    let payload = JSON.parse(msg);
                    if (payload.id !== 'get_source_code') {{
                        alert('invalid calling order. you need to set the source code first');
                        return;
                    }}
                    window.eval(payload.response);
                    window.{self.app_js_render_fn}('{self.app_id}', tunnel);
                }});
                tunnel.send_msg(JSON.stringify({{ url: '/get_source_code', params: null, id: 'get_source_code' }}));
                return true;
            }}
            repeatUntilSuccess(setupSliderApp, 50, 10);
            """))
        return self
    
    def set_data(self, lst: List[TypedDict('SliderAppLst', description=str, args=Any)], start_index: int=0):
        self.lst = lst
        self.index = start_index
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "wait_for_client_ready"
            },
            {
                "type": "set_props",
                "props": {
                    "min": 0,
                    "max": len(lst) - 1,
                    "index": self.index,
                    "description": lst[self.index]['description']
                }
            }
        ]).decode())
        self.app_render_fn(*lst[self.index]['args'])

    @_SliderApp.register_handler("/view")
    def view(self, params: dict):
        self.index = params['index']
        self.tunnel.send_msg(orjson.dumps([
            {
                "type": "set_props",
                "props": {
                    "description": self.lst[self.index]['description']
                }
            }
        ]).decode())
        self.app_render_fn(*self.lst[self.index]['args'])


class BasicApp:
    """A basic application to make default jupyter application work with slider"""
    def __init__(self, render_fn: Callable[[Any], None]):
        self.output = ipywidgets.Output()
        self.render_fn = render_fn
        self.app_id = str(uuid4())

    def render(self, same_tab: bool = True, new_window: bool = False, shadow_dom: Optional[bool] = None):
        assert same_tab is True and new_window is False
        display(HTML(f'<div id="{self.app_id}"></div>'), self.output)

    def set_data(self, item):
        with self.output:
            self.output.clear_output()
            self.render_fn(item)