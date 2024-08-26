import os
import dearpygui.dearpygui as dpg
import numpy as np
import torch

class Context:
    def __init__(self, current, segments, features) -> None:
        self.segments = segments
        self.features = features
        self.set_current(current)

    def set_current(self, current):
        self.current = current
        self.cs = torch.from_numpy(np.load(self.segments[current]))
        self.cf = torch.from_numpy(np.load(self.features[current]))

    def get_color_map(self, level=3):
        segment = self.cs[level]
        feature = self.cf
        mask = segment!=-1
        fmap=feature[segment.long()]
        return (fmap+1)/2*mask.unsqueeze(-1)
    
    def get_value(self, x, y): # x:985, y:725
        fmap=self.cf[self.cs[3].long()]
        self.value = fmap[y][x]
        print(self.value)

    def set_value(self, x, y):
        self.cf[self.cs[3][y][x].long()] = self.value
        dpg.set_value('_texture', self.get_color_map().numpy())
        print(self.value)

    def save(self):
        np.save(os.path.join('data/lerf_ovs/waldo_kitchen/temp', f'frame_{str(self.current).rjust(5,'0')}_s.npy'), self.cs)
        np.save(os.path.join('data/lerf_ovs/waldo_kitchen/temp', f'frame_{str(self.current).rjust(5,'0')}_f.npy'), self.cf)

def main():
    mask_dir = 'data/lerf_ovs/waldo_kitchen/temp' #'data/lerf_ovs/waldo_kitchen/language_features_dim3'
    feature_level = 3
    width = 985
    height = 725
    render_buffer = np.zeros((width, height, 3), dtype=np.float32)

    current = 0
    segments = [os.path.join(mask_dir,p) for p in os.listdir(mask_dir) if p.endswith('s.npy')]
    features = [os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.endswith('f.npy')]
    segments.sort()
    features.sort()

    context = Context(current, segments, features)
    
    dpg.create_context()
    ### register texture
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(width, height, render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

    ### register window
    with dpg.window(tag="_primary_window", width=width+350, height=height+50):
        dpg.add_image("_texture", pos=(0,20))   # add the texture

    dpg.set_primary_window("_primary_window", True)

    def button_callback(sender, app_data, user_data):
        context: Context = user_data[0]
        offset = user_data[1]
        context.set_current(context.current+offset)
        cmap=context.get_color_map()
        dpg.set_value('_texture', cmap.numpy())
        dpg.set_value('index', f'index: {context.current}')
    def get_callback(sender, app_data, context:Context):
        x=int(float(dpg.get_value('x')[3:]))
        y=int(float(dpg.get_value('y')[3:]))
        context.get_value(x,y)
    def set_callback(sender, app_data, context:Context):
        x=int(float(dpg.get_value('x')[3:]))
        y=int(float(dpg.get_value('y')[3:]))
        value=context.set_value(x,y)
    def save_callback(sender, app_data, context:Context):
        context.save()
    with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[width+30, 0]):
        dpg.add_text("\noption: ", tag="option")
        dpg.add_text("x: ", tag="x")
        dpg.add_text("y: ", tag="y")
        dpg.add_text('index: ', tag='index')
        dpg.add_button(label='prev', callback=button_callback, user_data=(context, -1))
        dpg.add_button(label='next', callback=button_callback, user_data=(context, 1))
        dpg.add_button(label='get', callback=get_callback, user_data=context)
        dpg.add_button(label='set', callback=set_callback, user_data=context)
        dpg.add_button(label='save', callback=save_callback, user_data=context)

    def mouse_click_handler(sender, app_data, user_data):
        print(sender, app_data)
        pos = dpg.get_mouse_pos()
        if pos[0]<=985 and pos[0]>=0 and pos[1]<=725 and pos[1]>=0:
            dpg.set_value('x', f'x: {pos[0]}')
            dpg.set_value('y', f'y: {pos[1]}')

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=mouse_click_handler)


    dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=width+350, height=height+50, resizable=False)

    ### global theme
    with dpg.theme() as theme_no_padding:
        with dpg.theme_component(dpg.mvAll):
            # set all padding to 0 to avoid scroll bar
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
    dpg.bind_item_theme("_primary_window", theme_no_padding)

    dpg.setup_dearpygui()

    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

main()