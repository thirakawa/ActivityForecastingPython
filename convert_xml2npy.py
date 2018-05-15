# -*- coding: utf-8 -*-

import os
import numpy as np
import xml.dom.minidom
import glob


def load_opencv_xml(filename):
    dom = xml.dom.minidom.parse(filename)
    mat_list = []
    for node in dom.documentElement.childNodes:
        # each features
        if node.nodeType == node.ELEMENT_NODE:
            # print node.tagName
            # rows, cols, data type, data
            for node2 in node.childNodes:
                if node2.nodeType == node2.ELEMENT_NODE:
                    # stack data
                    for node3 in node2.childNodes:
                        if node3.nodeType == node3.TEXT_NODE:
                            if node2.tagName == 'rows':
                                rows = int(node3.data)
                            elif node2.tagName == 'cols':
                                cols = int(node3.data)
                            elif node2.tagName == 'data':
                                s = node3.data
                                s = s.replace('\n', '')
                                s = s.replace('   ', '')
                                s = s.split(' ')
                                s.remove('')
                                fvec = map(float, s)
                                mat = np.array(fvec).reshape([rows, cols])
                                mat_list.append(mat)
    return np.array(mat_list)


if __name__ == '__main__':

    xml_file_prefix = "ioc_demo/walk_feat/"
    filenames = glob.glob(xml_file_prefix + "*.xml")

    for fname in filenames:
        print fname
        feature_mat = load_opencv_xml(fname)
        base, ext = os.path.splitext(fname)
        np.save(base + ".npy", feature_mat)
    print "done!"