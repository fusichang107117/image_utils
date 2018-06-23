
# net_struct = {
#     'alexnet':  {'net':  [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]],
#                  'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']},
#     'vgg16':    {'net':  [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
#                          [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
#                          [2, 2, 0]],
#                  'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
#                           'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
#                           'pool5']},
#     'zf-5':     {'net':  [[7, 2, 3], [3, 2, 1], [5, 2, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
#                  'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}
# }

imsize = 300

net_struct = {
    'vgg16_ssd_conv4_3': {'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_3_mbox']
                          },
    'vgg16_ssd_fc7':     {
                          'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [3, 1, 1], [13, 1, 6], [1, 1, 0], [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
                                   'pool5',   'fc_6',  'fc_7',    'fc_7_mbox']
                         },
    'vgg16_ssd_conv6_2': {
                          'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [3, 1, 1], [13, 1, 6], [1, 1, 0], [1, 1, 0], [3, 2, 1], [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
                                   'pool5',   'fc_6',  'fc_7',    'conv6_1', 'conv6_2', 'conv6_2_mbox']
                          },
    'vgg16_ssd_conv7_2': {
                          'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [3, 1, 1], [13, 1, 6], [1, 1, 0], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1],
                                  [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',   'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',   'conv5_1', 'conv5_2', 'conv5_3',
                                   'pool5',   'fc_6',  'fc_7',    'conv6_1', 'conv6_2', 'conv7_1', 'conv7_2', 'conv7_2_mbox']
                         },
    'vgg16_ssd_conv8_2': {
                          'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [3, 1, 1], [13, 1, 6], [1, 1, 0], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1],
                                  [1, 1, 0], [3, 1, 0], [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',   'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',   'conv5_1', 'conv5_2', 'conv5_3',
                                   'pool5',   'fc_6',  'fc_7',    'conv6_1', 'conv6_2', 'conv7_1', 'conv7_2', 'conv8_1', 'conv8_2',
                                   'conv8_2_mbox']
                         },
    'vgg16_ssd_conv9_2': {
                          'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                                  [3, 1, 1], [13, 1, 6], [1, 1, 0], [1, 1, 0], [3, 2, 1], [1, 1, 0], [3, 2, 1],
                                  [1, 1, 0], [3, 1, 0], [1, 1, 0], [3, 1, 0], [3, 1, 1]],
                          'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2',   'conv3_1', 'conv3_2', 'conv3_3',
                                   'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4',   'conv5_1', 'conv5_2', 'conv5_3',
                                   'pool5',   'fc_6',  'fc_7',    'conv6_1', 'conv6_2', 'conv7_1', 'conv7_2', 'conv8_1', 'conv8_2',
                                   'conv9_1', 'conv9_2', "conv9_2_mbox"]
                         },

}


def top_from_bottom(imsize, net, layernum):
    total_stride = 1
    insize = imsize
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        total_stride = total_stride * stride
    return outsize, total_stride


def bottom_from_top(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))

    for net in net_struct.keys():
        print('************net structrue name is %s**************' % net)
        for i in range(len(net_struct[net]['net'])):
            p = top_from_bottom(imsize, net_struct[net]['net'], i + 1)
            rf = bottom_from_top(net_struct[net]['net'], i + 1)
            print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (net_struct[net]['name'][i], p[0], p[1], rf))