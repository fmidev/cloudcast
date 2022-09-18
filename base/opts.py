class CloudCastOptions():
    def __init__(self, **kwargs):
        try:
            label = kwargs['label']
            if label is not None:
                self.from_label(label)
            else:
                raise KeyError("")
        except KeyError as e:
            self.model = kwargs.get('model', 'unet')
            self.n_channels = kwargs.get('n_channels', 1)
            self.loss_function = kwargs.get('loss_function', 'MeanSquaredError')
            self.preprocess = kwargs.get('preprocess', 'img_size=128x128')
            self.include_datetime = kwargs.get('include_datetime', False)
            self.include_topography = kwargs.get('include_topography', False)
            self.include_terrain_type = kwargs.get('include_terrain_type', False)
            self.onehot_encoding = kwargs.get('onehot_encoding', False)
            self.leadtime_conditioning = kwargs.get('leadtime_conditioning', 0)

    def __str__(self):
        return self.get_label()

    def from_label(self, label):
        elems = label.split('-')

        assert(len(elems) == 9)

        self.model = elems[0]
        self.loss_function = elems[1]
        self.n_channels = int(elems[2].split('=')[1])
        self.include_datetime = eval(elems[3].split('=')[1])
        self.include_topography = eval(elems[4].split('=')[1])
        self.include_terrain_type = eval(elems[5].split('=')[1])
        self.leadtime_conditioning = int(elems[6].split('=')[1])
        self.onehot_encoding = eval(elems[7].split('=')[1])
        self.preprocess = elems[8]


    def get_label(self):
        return "{}-{}-hist={}-dt={}-topo={}-terrain={}-lc={}-oh={}-{}".format(
                             self.model,
                             self.loss_function,
                             self.n_channels,
                             self.include_datetime,
                             self.include_topography,
                             self.include_terrain_type,
                             self.leadtime_conditioning,
                             self.onehot_encoding,
                             self.preprocess)

    def get_reduced_label(self):
        return get_label().replace('True','T') \
                   .replace('False','F') \
                   .replace('binary_crossentropy', 'bc') \
                   .replace('MeanSquaredError', 'MSE') \
                   .replace('MeanAbsoluteError', 'MAE') \
                   .replace('img_size', 'i_s') \
                   .replace('terrain', 'tr')

