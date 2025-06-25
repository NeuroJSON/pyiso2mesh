import unittest
import numpy as np

import sys
import os

# Add project_root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iso2mesh import *
import matplotlib

mpl_ver = matplotlib.__version__.split(".")


class Test_geometry(unittest.TestCase):
    def test_meshabox(self):
        no, fc, el = meshabox([0, 0, 0], [1, 1, 1], 1, method="tetgen1.5")
        expected_fc = [
            [1, 3, 2],
            [6, 1, 2],
            [1, 4, 3],
            [4, 1, 8],
            [1, 6, 5],
            [8, 1, 5],
            [7, 2, 3],
            [6, 2, 7],
            [4, 8, 3],
            [8, 7, 3],
            [6, 7, 5],
            [8, 5, 7],
        ]
        self.assertEqual(fc.tolist(), expected_fc)
        self.assertEqual(round(sum(elemvolume(no, el)), 2), 1)

    def test_meshunitsphere(self):
        no, fc, el = meshunitsphere(0.05, 100)
        self.assertAlmostEqual(round(sum(elemvolume(no, fc[:, :3])), 3), 12.553)
        self.assertAlmostEqual(round(sum(elemvolume(no, el[:, :4])), 3), 4.180)

    def test_meshasphere(self):
        no, fc, el = meshasphere([1, 1, 1], 2, 0.1, 100)
        self.assertAlmostEqual(round(sum(elemvolume(no, fc[:, :3])), 1), 50.2)
        self.assertAlmostEqual(round(sum(elemvolume(no, el[:, :4])), 1), 33.4)

    def test_meshgrid5(self):
        no, el = meshgrid5(np.arange(1, 3), np.arange(-1, 1), np.arange(2, 4))
        self.assertEqual(sum(el).tolist(), [545, 577, 532, 586])
        self.assertEqual(
            el[:, 3].tolist(),
            [
                13,
                5,
                11,
                14,
                5,
                5,
                15,
                3,
                14,
                3,
                7,
                17,
                13,
                14,
                13,
                9,
                17,
                15,
                14,
                5,
                19,
                14,
                13,
                20,
                23,
                15,
                14,
                20,
                21,
                11,
                17,
                23,
                23,
                13,
                25,
                27,
                23,
                23,
                15,
                15,
            ],
        )

    def test_meshgrid6(self):
        no, el = meshgrid6(np.arange(1, 3), np.arange(-1, 1), np.arange(2, 3.5, 0.5))
        expected = [
            [1, 2, 8, 4],
            [5, 6, 12, 8],
            [1, 3, 4, 8],
            [5, 7, 8, 12],
            [1, 2, 6, 8],
            [5, 6, 10, 12],
            [1, 5, 8, 6],
            [5, 9, 12, 10],
            [1, 3, 8, 7],
            [5, 7, 12, 11],
            [1, 5, 7, 8],
            [5, 9, 11, 12],
        ]
        self.assertEqual(el.tolist(), expected)

    def test_latticegrid(self):
        no, fc, c0 = latticegrid(
            np.arange(1, 3), np.arange(-1, 1), np.arange(2, 3.5, 0.5)
        )
        expected_fc = [
            [1, 2, 6, 5],
            [1, 3, 4, 2],
            [1, 5, 7, 3],
            [2, 6, 8, 4],
            [3, 4, 8, 7],
            [5, 6, 10, 9],
            [5, 7, 8, 6],
            [5, 9, 11, 7],
            [6, 10, 12, 8],
            [7, 8, 12, 11],
            [9, 11, 12, 10],
        ]
        expected_c0 = [[1.5, -0.5, 2.25], [1.5, -0.5, 2.75]]
        self.assertEqual(fc, expected_fc)
        self.assertEqual(np.round(c0, 6).tolist(), expected_c0)

    def test_meshanellip(self):
        no, fc, el = meshanellip([1, 1, 1], [2, 4, 1], 0.05, 100)
        self.assertEqual(round(sum(elemvolume(no, fc[:, :3])), 2), 63.41)
        self.assertEqual(round(sum(elemvolume(no, el[:, :4])), 2), 33.44)

    def test_meshacylinder_plc(self):
        (
            no,
            fc,
        ) = meshacylinder([1, 1, 1], [2, 3, 4], [0.5, 0.8], 0, 0, 8)
        expected_fc = [
            [[[1, 9, 10, 2]], [1]],
            [[[2, 10, 11, 3]], [1]],
            [[[3, 11, 12, 4]], [1]],
            [[[4, 12, 13, 5]], [1]],
            [[[5, 13, 14, 6]], [1]],
            [[[6, 14, 15, 7]], [1]],
            [[[7, 15, 16, 8]], [1]],
            [[[8, 16, 9, 1]], [1]],
            [[[1, 2, 3, 4, 5, 6, 7, 8]], [2]],
            [[[9, 10, 11, 12, 13, 14, 15, 16]], [3]],
        ]
        self.assertEqual(fc, expected_fc)

    def test_meshacylinder(self):
        no, fc, el = meshacylinder([1, 1, 1], [2, 3, 4], [10, 12], 0.1, 10)
        self.assertEqual(round(sum(elemvolume(no, fc)), 4), 1045.2322)
        self.assertEqual(round(sum(elemvolume(no, el[:, :4])), 4), 1402.8993)

    def test_meshacylinders_plc(self):
        (
            no,
            fc,
        ) = meshcylinders([10, 20, 30], [0, 1, 1], [2, 4], 3, 0, 0, 8)
        expected_fc = [
            [[[18, 19, 14, 12]], [1]],
            [[[12, 14, 7, 6]], [1]],
            [[[6, 7, 2, 1]], [1]],
            [[[1, 2, 5, 4]], [1]],
            [[[4, 5, 11, 10]], [1]],
            [[[10, 11, 17, 16]], [1]],
            [[[16, 17, 23, 22]], [1]],
            [[[22, 23, 19, 18]], [1]],
            [[[18, 12, 6, 1, 4, 10, 16, 22]], [2]],
            [[[19, 14, 7, 2, 5, 11, 17, 23]], [3]],
            [[[19, 21, 15, 14]], [1]],
            [[[14, 15, 9, 7]], [1]],
            [[[7, 9, 3, 2]], [1]],
            [[[2, 3, 8, 5]], [1]],
            [[[5, 8, 13, 11]], [1]],
            [[[11, 13, 20, 17]], [1]],
            [[[17, 20, 24, 23]], [1]],
            [[[23, 24, 21, 19]], [1]],
            [[[21, 15, 9, 3, 8, 13, 20, 24]], [3]],
        ]
        self.assertEqual(fc, expected_fc)
        self.assertEqual(np.sum(no), 1568)

    def test_meshacylinders(self):
        no, fc, el = meshcylinders([10, 20, 30], [0, 1, 1], [2, 4], 3, 1, 10, 8)
        self.assertAlmostEqual(
            sum(elemvolume(no, fc[fc[:, -1] == 1, :3])), 155.86447684358734, 3
        )
        self.assertAlmostEqual(
            sum(elemvolume(no, el[el[:, -1] == 2, :4])), 144.0000000027395, 3
        )


class Test_trait(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_trait, self).__init__(*args, **kwargs)
        self.no, self.el = meshgrid6(
            np.arange(1, 3), np.arange(-1, 1), np.arange(2, 3.5, 0.5)
        )
        self.fc = volface(self.el)[0]

    def test_volface(self):
        expected_fc = [
            [2, 1, 4],
            [1, 2, 6],
            [1, 3, 4],
            [3, 1, 7],
            [5, 1, 6],
            [1, 5, 7],
            [2, 4, 8],
            [2, 8, 6],
            [3, 8, 4],
            [3, 7, 8],
            [5, 6, 10],
            [7, 5, 11],
            [9, 5, 10],
            [5, 9, 11],
            [6, 8, 12],
            [6, 12, 10],
            [7, 12, 8],
            [7, 11, 12],
            [9, 10, 12],
            [9, 12, 11],
        ]
        self.assertEqual(self.fc.tolist(), expected_fc)

    def test_uniqfaces(self):
        output = uniqfaces(self.el)[0]
        expected = [
            [1, 2, 4],
            [1, 2, 6],
            [1, 2, 8],
            [1, 3, 4],
            [1, 3, 7],
            [1, 3, 8],
            [1, 8, 4],
            [1, 5, 6],
            [1, 5, 7],
            [1, 5, 8],
            [1, 6, 8],
            [1, 8, 7],
            [2, 8, 4],
            [2, 6, 8],
            [3, 4, 8],
            [3, 8, 7],
            [5, 6, 8],
            [5, 6, 10],
            [5, 6, 12],
            [5, 7, 8],
            [5, 7, 11],
            [5, 7, 12],
            [5, 12, 8],
            [5, 9, 10],
            [5, 9, 11],
            [5, 9, 12],
            [5, 10, 12],
            [5, 12, 11],
            [6, 12, 8],
            [6, 10, 12],
            [7, 8, 12],
            [7, 12, 11],
            [9, 12, 10],
            [9, 11, 12],
        ]
        self.assertEqual(output.tolist(), expected)

    def test_meshedge(self):
        output = np.unique(meshedge(self.el), axis=0)
        expected = [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [1, 7],
            [1, 8],
            [2, 4],
            [2, 6],
            [2, 8],
            [3, 4],
            [3, 7],
            [3, 8],
            [4, 8],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 9],
            [5, 10],
            [5, 11],
            [5, 12],
            [6, 8],
            [6, 10],
            [6, 12],
            [7, 8],
            [7, 11],
            [7, 12],
            [8, 4],
            [8, 6],
            [8, 7],
            [8, 12],
            [9, 10],
            [9, 11],
            [9, 12],
            [10, 12],
            [11, 12],
            [12, 8],
            [12, 10],
            [12, 11],
        ]
        self.assertEqual(output.tolist(), expected)

    def test_meshface(self):
        output = meshface(self.el)
        expected = [
            [1, 2, 8],
            [5, 6, 12],
            [1, 3, 4],
            [5, 7, 8],
            [1, 2, 6],
            [5, 6, 10],
            [1, 5, 8],
            [5, 9, 12],
            [1, 3, 8],
            [5, 7, 12],
            [1, 5, 7],
            [5, 9, 11],
            [1, 2, 4],
            [5, 6, 8],
            [1, 3, 8],
            [5, 7, 12],
            [1, 2, 8],
            [5, 6, 12],
            [1, 5, 6],
            [5, 9, 10],
            [1, 3, 7],
            [5, 7, 11],
            [1, 5, 8],
            [5, 9, 12],
            [1, 8, 4],
            [5, 12, 8],
            [1, 4, 8],
            [5, 8, 12],
            [1, 6, 8],
            [5, 10, 12],
            [1, 8, 6],
            [5, 12, 10],
            [1, 8, 7],
            [5, 12, 11],
            [1, 7, 8],
            [5, 11, 12],
            [2, 8, 4],
            [6, 12, 8],
            [3, 4, 8],
            [7, 8, 12],
            [2, 6, 8],
            [6, 10, 12],
            [5, 8, 6],
            [9, 12, 10],
            [3, 8, 7],
            [7, 12, 11],
            [5, 7, 8],
            [9, 11, 12],
        ]
        self.assertEqual(output.tolist(), expected)

    def test_uniqedges(self):
        ed, idx, newel = uniqedges(self.el)
        expected_ed = [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [1, 7],
            [1, 8],
            [2, 4],
            [2, 6],
            [2, 8],
            [3, 4],
            [3, 7],
            [3, 8],
            [8, 4],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 9],
            [5, 10],
            [5, 11],
            [5, 12],
            [6, 8],
            [6, 10],
            [6, 12],
            [7, 8],
            [7, 11],
            [7, 12],
            [12, 8],
            [9, 10],
            [9, 11],
            [9, 12],
            [10, 12],
            [12, 11],
        ]
        expected_el = [
            [1, 7, 3, 10, 8, 14],
            [15, 21, 17, 24, 22, 28],
            [2, 3, 7, 11, 13, 14],
            [16, 17, 21, 25, 27, 28],
            [1, 5, 7, 9, 10, 22],
            [15, 19, 21, 23, 24, 32],
            [4, 7, 5, 17, 15, 22],
            [18, 21, 19, 31, 29, 32],
            [2, 7, 6, 13, 12, 25],
            [16, 21, 20, 27, 26, 33],
            [4, 6, 7, 16, 17, 25],
            [18, 20, 21, 30, 31, 33],
        ]
        self.assertEqual(ed.tolist(), expected_ed)
        self.assertEqual(
            idx.tolist(),
            [
                1,
                3,
                15,
                7,
                17,
                23,
                13,
                49,
                41,
                37,
                39,
                57,
                45,
                61,
                2,
                4,
                16,
                8,
                18,
                24,
                14,
                50,
                42,
                38,
                40,
                58,
                46,
                62,
                56,
                48,
                44,
                66,
                70,
            ],
        )
        self.assertEqual(newel.tolist(), expected_el)

    def test_meshconn(self):
        expected = [
            [2, 3, 4, 5, 6, 7, 8],
            [1, 4, 6, 8],
            [1, 4, 7, 8],
            [1, 2, 3, 8],
            [1, 6, 7, 8, 9, 10, 11, 12],
            [1, 2, 5, 8, 10, 12],
            [1, 3, 5, 8, 11, 12],
            [1, 2, 3, 4, 5, 6, 7, 12],
            [5, 10, 11, 12],
            [5, 6, 9, 12],
            [5, 7, 9, 12],
            [5, 6, 7, 8, 9, 10, 11],
        ]
        self.assertEqual(meshconn(self.el, self.no.shape[0])[0], expected)

    def test_mesheuler(self):
        eu = mesheuler(self.el)
        self.assertEqual(eu, (1, 12, 33, 34, 0, 0, 12))

        eu = mesheuler(self.fc)
        self.assertEqual(eu, (2, 12, 30, 20, 0, 0, 0))

        eu = mesheuler(self.fc[1:-2, :])
        self.assertEqual(eu, (0, 12, 29, 17, 2, 0, 0))

    def test_neighborelem(self):
        expected = [
            [1, 3, 5, 7, 9, 11],
            [1, 5],
            [3, 9],
            [1, 3],
            [2, 4, 6, 7, 8, 10, 11, 12],
            [2, 5, 6, 7],
            [4, 9, 10, 11],
            [1, 2, 3, 4, 5, 7, 9, 11],
            [8, 12],
            [6, 8],
            [10, 12],
            [2, 4, 6, 8, 10, 12],
        ]
        self.assertEqual(neighborelem(self.el, self.no.shape[0])[0], expected)

    def test_faceneighbors(self):
        expected = [
            [5, 0, 3, 0],
            [6, 7, 4, 0],
            [0, 9, 1, 0],
            [11, 10, 2, 0],
            [0, 1, 7, 0],
            [0, 2, 8, 0],
            [11, 0, 5, 2],
            [12, 0, 6, 0],
            [3, 0, 11, 0],
            [4, 0, 12, 0],
            [0, 7, 9, 4],
            [0, 8, 10, 0],
        ]
        self.assertEqual(faceneighbors(self.el).tolist(), expected)

    def test_faceneighbors_surface(self):
        expected = [
            [1, 3, 4],
            [1, 2, 6],
            [5, 6, 10],
            [1, 5, 7],
            [5, 9, 11],
            [1, 2, 4],
            [1, 5, 6],
            [5, 9, 10],
            [1, 3, 7],
            [5, 7, 11],
            [2, 4, 8],
            [6, 8, 12],
            [3, 4, 8],
            [7, 8, 12],
            [2, 6, 8],
            [6, 10, 12],
            [9, 10, 12],
            [3, 7, 8],
            [7, 11, 12],
            [9, 11, 12],
        ]
        self.assertEqual(faceneighbors(self.el, "surface").tolist(), expected)

    def test_edgeneighbors(self):
        expected = [
            [2, 3, 7],
            [1, 8, 5],
            [4, 9, 1],
            [3, 6, 10],
            [6, 2, 11],
            [5, 12, 4],
            [1, 9, 8],
            [7, 15, 2],
            [10, 7, 3],
            [4, 17, 9],
            [5, 16, 13],
            [6, 14, 18],
            [14, 11, 19],
            [13, 20, 12],
            [8, 17, 16],
            [15, 19, 11],
            [18, 15, 10],
            [12, 20, 17],
            [13, 16, 20],
            [19, 18, 14],
        ]
        self.assertEqual(edgeneighbors(self.fc).tolist(), expected)

    def test_meshquality(self):
        expected = [0.620322, 0.686786]
        result = np.unique(meshquality(self.no, self.el).round(decimals=6))
        self.assertEqual(result.tolist(), expected)

    def test_meshquality_face(self):
        expected = [0.69282, 0.866025]
        result = np.unique(meshquality(self.no, self.fc).round(decimals=6))
        self.assertEqual(result.tolist(), expected)

    def test_meshcentroid_face(self):
        expected = [
            [1.667, -0.667, 2],
            [1.667, -1, 2.167],
            [1.333, -0.333, 2],
            [1, -0.333, 2.167],
            [1.333, -1, 2.333],
            [1, -0.667, 2.333],
            [2, -0.333, 2.167],
            [2, -0.667, 2.333],
            [1.667, 0, 2.167],
            [1.333, 0, 2.333],
            [1.667, -1, 2.667],
            [1, -0.333, 2.667],
            [1.333, -1, 2.833],
            [1, -0.667, 2.833],
            [2, -0.333, 2.667],
            [2, -0.667, 2.833],
            [1.667, 0, 2.667],
            [1.333, 0, 2.833],
            [1.667, -0.667, 3],
            [1.333, -0.333, 3],
        ]
        result = meshcentroid(self.no, self.fc).round(decimals=3).tolist()
        self.assertEqual(result, expected)

    def test_elemvolume(self):
        expected = [0.083333]
        result = np.unique(elemvolume(self.no, self.el).round(decimals=6)).tolist()
        self.assertEqual(result, expected)

    def test_nodevolume(self):
        expected = np.sum(elemvolume(self.no, self.el))
        result = np.sum(nodevolume(self.no, self.el))
        self.assertAlmostEqual(result, expected, 7)

    def test_surfvolume(self):
        self.assertEqual(surfvolume(self.no, self.fc), 1)

    def test_surfacenorm(self):
        snorm = surfacenorm(self.no, self.fc)
        expected = [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
        self.assertEqual(snorm.tolist(), expected)

    def test_nodesurfnorm(self):
        nnorm = nodesurfnorm(self.no, self.fc)
        expected = [
            [-0.57735, -0.57735, -0.57735],
            [0.816497, -0.408248, -0.408248],
            [-0.408248, 0.816497, -0.408248],
            [0.408248, 0.408248, -0.816497],
            [-0.707107, -0.707107, 0.0],
            [0.707107, -0.707107, 0.0],
            [-0.707107, 0.707107, 0.0],
            [0.707107, 0.707107, 0.0],
            [-0.408248, -0.408248, 0.816497],
            [0.408248, -0.816497, 0.408248],
            [-0.816497, 0.408248, 0.408248],
            [0.57735, 0.57735, 0.57735],
        ]
        self.assertEqual(nnorm.round(6).tolist(), expected)

    def test_insurface(self):
        pts = np.array([[1.5, -0.9, 2.1], [1, 0, 2], [-1, 0, 2], [1.2, 0, 2.5]])
        expected = [1, 1, 0, 1]
        result = insurface(self.no, self.fc, pts).tolist()
        self.assertEqual(result, expected)

    def test_highordertet(self):
        no1, el1 = highordertet(self.no, self.el)
        expected_el = [
            [2, 8, 4, 11, 9, 15],
            [16, 22, 18, 25, 23, 29],
            [3, 4, 8, 12, 14, 15],
            [17, 18, 22, 26, 28, 29],
            [2, 6, 8, 10, 11, 23],
            [16, 20, 22, 24, 25, 33],
            [5, 8, 6, 18, 16, 23],
            [19, 22, 20, 32, 30, 33],
            [3, 8, 7, 14, 13, 26],
            [17, 22, 21, 28, 27, 34],
            [5, 7, 8, 17, 18, 26],
            [19, 21, 22, 31, 32, 34],
        ]
        self.assertEqual(el1.tolist(), expected_el)
        self.assertEqual(np.sum(no1), 115.5)

    def test_elemfacecenter(self):
        no1, el1 = elemfacecenter(self.no, self.el)
        expected_el = [
            [3, 1, 7, 13],
            [19, 17, 23, 29],
            [4, 6, 7, 15],
            [20, 22, 23, 31],
            [2, 3, 11, 14],
            [18, 19, 27, 30],
            [10, 8, 11, 17],
            [26, 24, 27, 33],
            [6, 5, 12, 16],
            [22, 21, 28, 32],
            [9, 10, 12, 20],
            [25, 26, 28, 34],
        ]
        self.assertEqual(el1.tolist(), expected_el)
        self.assertEqual(np.sum(no1), 119.0)

    def test_barydualmesh(self):
        no1, el1 = barydualmesh(self.no, self.el)
        expected_el = [
            [2, 34, 68, 36],
            [16, 50, 69, 52],
            [3, 39, 70, 37],
            [17, 55, 71, 53],
            [2, 36, 72, 35],
            [16, 52, 73, 51],
            [5, 41, 74, 43],
            [19, 57, 75, 59],
            [3, 38, 76, 39],
            [17, 54, 77, 55],
            [5, 43, 78, 42],
            [19, 59, 79, 58],
            [8, 36, 68, 40],
            [22, 52, 69, 56],
            [4, 37, 70, 40],
            [18, 53, 71, 56],
            [6, 35, 72, 44],
            [20, 51, 73, 60],
            [8, 43, 74, 44],
            [22, 59, 75, 60],
            [8, 39, 76, 45],
            [22, 55, 77, 61],
            [7, 42, 78, 45],
            [21, 58, 79, 61],
            [4, 40, 68, 34],
            [18, 56, 69, 50],
            [8, 40, 70, 39],
            [22, 56, 71, 55],
            [8, 44, 72, 36],
            [22, 60, 73, 52],
            [6, 44, 74, 41],
            [20, 60, 75, 57],
            [7, 45, 76, 38],
            [21, 61, 77, 54],
            [8, 45, 78, 43],
            [22, 61, 79, 59],
            [11, 36, 68, 46],
            [25, 52, 69, 62],
            [12, 37, 70, 48],
            [26, 53, 71, 64],
            [10, 35, 72, 47],
            [24, 51, 73, 63],
            [18, 43, 74, 50],
            [32, 59, 75, 66],
            [14, 39, 76, 49],
            [28, 55, 77, 65],
            [17, 42, 78, 53],
            [31, 58, 79, 67],
            [9, 34, 68, 46],
            [23, 50, 69, 62],
            [14, 39, 70, 48],
            [28, 55, 71, 64],
            [11, 36, 72, 47],
            [25, 52, 73, 63],
            [16, 41, 74, 50],
            [30, 57, 75, 66],
            [13, 38, 76, 49],
            [27, 54, 77, 65],
            [18, 43, 78, 53],
            [32, 59, 79, 67],
            [15, 40, 68, 46],
            [29, 56, 69, 62],
            [15, 40, 70, 48],
            [29, 56, 71, 64],
            [23, 44, 72, 47],
            [33, 60, 73, 63],
            [23, 44, 74, 50],
            [33, 60, 75, 66],
            [26, 45, 76, 49],
            [34, 61, 77, 65],
            [26, 45, 78, 53],
            [34, 61, 79, 67],
        ]
        self.assertEqual(el1.tolist(), expected_el)
        self.assertEqual(np.sum(no1), 276.5)


class Test_modify(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_modify, self).__init__(*args, **kwargs)
        self.no, self.fc, self.el = meshacylinder(
            [1, 1, 1], [2, 3, 4], [0.5, 0.7], 1, 100, 8
        )
        self.el = meshreorient(self.no, self.el[:, :4])[0]
        self.no1, self.fc1 = meshcheckrepair(self.no, self.fc, "deep")
        self.nbox, self.ebox = meshgrid5([0, 1], [1, 2], [0, 2])
        self.fbox = volface(self.ebox)[0]

    def test_removeisolatednode(self):
        no1 = removeisolatednode(self.no[:, :3], self.fc[:, :3])[0]
        self.assertTrue(no1.shape[0], np.unique(self.fc[:, :3]).size)

    def test_meshreorient(self):
        self.assertEqual(
            meshreorient(self.no, self.el[:, [0, 1, 3, 2]])[0].tolist(),
            meshreorient(self.no, self.el[:, :4])[0].tolist(),
        )

    def test_removedupnodes(self):
        no1 = removedupnodes(np.vstack([self.no, self.no[:2, :]]), self.el[:, :4])[0]
        self.assertEqual(no1.shape, self.no.shape)

    def test_removedupelem(self):
        self.assertEqual(
            removedupelem(np.vstack([self.el[:, :4], self.el[:-5, :4]])).tolist(),
            self.el[-5:, :4].tolist(),
        )

    def test_surfreorient(self):
        no1, fc1 = surfreorient(self.no, self.fc)
        self.assertEqual(self.fc.shape, fc1.shape)
        self.assertFalse(np.any(elemvolume(no1, fc1) <= 0))

    def test_meshcheckrepair_deep(self):
        self.assertFalse(np.any(elemvolume(self.no1, self.fc1) <= 0))
        self.assertTrue(
            np.sum(elemvolume(self.no1, self.fc1)), np.sum(elemvolume(self.no, self.fc))
        )

    def test_meshcheckrepair_meshfix(self):
        no2, fc2 = meshcheckrepair(self.no, self.fc[3:, :], "meshfix")
        self.assertTrue(
            abs(sum(elemvolume(self.no1, self.fc1)) - sum(elemvolume(no2, fc2))) < 1e-4
        )

    def test_meshresample(self):
        no2, fc2 = meshresample(self.no1[:, :3], self.fc1[:, :3], 0.1)
        self.assertTrue(
            np.linalg.norm(
                no2[:6, :]
                - np.array(
                    [
                        [1.1988, 2.7369, 3.2748],
                        [1.2122, 1.6713, 1.7891],
                        [1.4297, 1.8648, 3.1964],
                        [1.8533, 2.6807, 4.2618],
                        [2.0276, 2.329, 2.6884],
                        [2.1149, 3.2916, 3.7673],
                    ]
                )
                < 0.01
            )
        )

    def test_sms(self):
        no, el = meshgrid5(np.arange(1, 3), np.arange(-1, 1), np.arange(2, 3.5, 0.5))
        no, el, _ = removeisolatednode(no, volface(el)[0])
        no1 = sms(no, el, 10)
        no2, el2, _ = s2m(no1, el, 1, 100)
        self.assertTrue(np.sum(elemvolume(no2, el2[:, :4])) > 0.8)

        no1 = sms(no, el, 10, 0.5, "laplacian")
        no2, el2, _ = s2m(no1, el, 1, 100)
        self.assertTrue(np.sum(elemvolume(no2, el2[:, :4])) < 0.1)

        no1 = sms(no, el, 10, 0.5, "lowpass")
        no2, el2, _ = s2m(no1, el, 1, 100)
        self.assertTrue(np.sum(elemvolume(no2, el2[:, :4])) > 0.55)

    def test_qmeshcut_elem(self):
        cutpos, cutvalue, facedata, _, _ = qmeshcut(
            self.ebox, self.nbox, self.nbox[:, 0] + self.nbox[:, 1], 2
        )
        expected_fc = [
            [1, 16, 29, 29],
            [2, 17, 30, 30],
            [3, 42, 55, 55],
            [4, 44, 57, 57],
            [5, 45, 58, 58],
            [19, 46, 71, 71],
            [6, 20, 32, 32],
            [7, 21, 33, 33],
            [35, 60, 72, 72],
            [36, 61, 73, 73],
            [8, 48, 62, 62],
            [24, 49, 75, 75],
            [37, 64, 76, 76],
            [10, 50, 65, 65],
            [12, 51, 67, 67],
            [39, 69, 78, 78],
            [13, 53, 70, 70],
            [14, 27, 40, 40],
            [15, 28, 54, 41],
            [18, 31, 56, 43],
            [22, 34, 59, 47],
            [9, 23, 74, 63],
            [11, 25, 77, 66],
            [26, 38, 68, 52],
        ]
        self.assertEqual(np.sum(cutpos), 234.0)
        self.assertEqual(facedata.tolist(), expected_fc)

    def test_qmeshcut_face(self):
        no2, fc2, _ = removeisolatednode(self.nbox, self.fbox)
        cutpos, cutvalue, facedata, _, _ = qmeshcut(fc2[:, :3], no2, no2[:, 0], 0)
        expected_fc = [
            [1, 22],
            [2, 12],
            [13, 23],
            [14, 24],
            [3, 15],
            [4, 25],
            [5, 16],
            [6, 26],
            [7, 27],
            [8, 17],
            [18, 28],
            [9, 19],
            [10, 29],
            [20, 30],
            [21, 31],
            [11, 32],
        ]
        self.assertEqual(np.sum(cutpos), 80.0)
        self.assertEqual(facedata.tolist(), expected_fc)

    def test_extractloops(self):
        no2, fc2, _ = removeisolatednode(self.nbox, self.fbox)
        cutpos, cutvalue, facedata, _, _ = qmeshcut(fc2[:, :3], no2, no2[:, 0], 0)
        _, ed2 = removedupnodes(cutpos, facedata)
        bcutloop = extractloops(ed2)

        expected_fc = [1, 4, 6, 7, 8, 5, 3, 2, 1]
        self.assertNotEqual(bcutloop[-1], bcutloop[-1])
        bcutloop.pop()
        self.assertEqual(bcutloop, expected_fc)


class Test_surfboolean(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_surfboolean, self).__init__(*args, **kwargs)

        self.no1, self.el1 = meshgrid5(
            np.arange(1, 3), np.arange(1, 3), np.arange(1, 3)
        )
        self.el1 = volface(self.el1)[0]
        self.no1, self.el1, _ = removeisolatednode(self.no1, self.el1)

        self.no2, self.el2 = meshgrid6(
            np.arange(1.7, 4), np.arange(1.7, 4), np.arange(1.7, 4)
        )
        self.el2 = volface(self.el2)[0]
        self.no2, self.el2, _ = removeisolatednode(self.no2, self.el2)

    def test_surfboolean_and(self):
        no3, el3 = surfboolean(self.no1, self.el1, "and", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(no3, el3, 1, 100)
        self.assertEqual(round(sum(elemvolume(node, elem[:, :4])), 5), 0.027)

    def test_surfboolean_or(self):
        no3, el3 = surfboolean(self.no1, self.el1, "or", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(no3, el3, 1, 100)
        self.assertEqual(round(sum(elemvolume(node, elem[:, :4])) * 1000), 8973)

    def test_surfboolean_diff(self):
        no3, el3 = surfboolean(self.no1, self.el1, "diff", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(no3, el3, 1, 100)
        self.assertEqual(round(sum(elemvolume(node, elem[:, :4])), 5), 0.973)

    def test_surfboolean_first(self):
        no3, el3 = surfboolean(self.no1, self.el1, "first", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(no3, el3, 1, 100, "tetgen", np.array([1.5, 1.5, 1.5]))
        self.assertEqual(
            round(sum(elemvolume(node, elem[elem[:, 4] == 1, :4])), 5), 0.973
        )
        self.assertEqual(
            round(sum(elemvolume(node, elem[elem[:, 4] == 0, :4])), 5), 0.027
        )

    def test_surfboolean_second(self):
        no3, el3 = surfboolean(self.no1, self.el1, "second", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(no3, el3, 1, 100, "tetgen", np.array([2.6, 2.6, 2.6]))
        self.assertAlmostEqual(
            sum(elemvolume(node, elem[elem[:, 4] == 1, :4])), 7.973, 5
        )
        self.assertAlmostEqual(
            sum(elemvolume(node, elem[elem[:, 4] == 0, :4])), 0.027, 5
        )

    def test_surfboolean_resolve(self):
        no3, el3 = surfboolean(self.no1, self.el1, "resolve", self.no2, self.el2)
        no3, el3 = meshcheckrepair(no3, el3, "dup", tolerance=1e-4)
        node, elem, _ = s2m(
            no3, el3, 1, 100, "tetgen", np.array([[1.5, 1.5, 1.5], [2.6, 2.6, 2.6]])
        )
        self.assertAlmostEqual(
            sum(elemvolume(node, elem[elem[:, 4] == 0, :4])), 0.027, 5
        )
        self.assertAlmostEqual(
            sum(elemvolume(node, elem[elem[:, 4] == 1, :4])), 0.973, 5
        )
        self.assertAlmostEqual(
            sum(elemvolume(node, elem[elem[:, 4] == 2, :4])), 7.973, 5
        )

    def test_surfboolean_self(self):
        self.assertEqual(
            surfboolean(self.no1, self.el1, "self", self.no2, self.el2)[0], 1
        )


class Test_core(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_core, self).__init__(*args, **kwargs)
        self.im = np.zeros((3, 3, 3))
        self.im[1, 1, 1:3] = 1

        xi, yi, zi = np.meshgrid(
            np.arange(0, 61), np.arange(0, 58), np.arange(0, 55), indexing="ij"
        )
        self.dist = (xi - 30) ** 2 + (yi - 30) ** 2 + (zi - 30) ** 2
        self.dist = self.dist.astype(float)
        self.mask = self.dist < 400
        self.mask = self.mask.astype(float)
        self.mask[25:35, 25:35, :] = 2

        self.plcno = (
            np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [1, 0, 0],
                    [0.5, 0.5, 1],
                    [0.2, 0.2, 0],
                    [0.2, 0.8, 0],
                    [0.8, 0.8, 0],
                    [0.8, 0.2, 0],
                ]
            )
            * 10
        )
        self.plcfc = [
            [1, 2, 5],
            [2, 3, 5],
            [3, 4, 5],
            [4, 1, 5],
            [1, 2, 3, 4, np.nan, 9, 8, 7, 6],
            [6, 7, 5],
            [7, 8, 5],
            [8, 9, 5],
            [9, 6, 5],
        ]

    def test_binsurface(self):
        no, fc = binsurface(self.im)
        expected_fc = [
            [10, 4, 1],
            [7, 10, 1],
            [12, 6, 5],
            [11, 5, 4],
            [2, 3, 9],
            [1, 2, 8],
            [11, 12, 5],
            [10, 11, 4],
            [2, 9, 8],
            [1, 8, 7],
            [8, 9, 12],
            [7, 8, 11],
            [6, 3, 2],
            [5, 2, 1],
            [8, 12, 11],
            [7, 11, 10],
            [5, 6, 2],
            [4, 5, 1],
        ]
        self.assertEqual(fc.tolist(), expected_fc)

        expected_sum = [3, 4, 5, 4, 5, 6, 4, 5, 6, 5, 6, 7]
        self.assertEqual(np.sum(no, axis=1).tolist(), expected_sum)

    def test_surfedge(self):
        no, fc = binsurface(self.im)
        expected_edge = [[6, 3], [3, 9], [12, 6], [9, 12]]
        self.assertEqual(surfedge(fc)[0].tolist(), expected_edge)

    def test_binsurface_4(self):
        no, fc = binsurface(self.im, 4)
        expected_quad = [
            [1, 3, 7, 5],
            [5, 7, 11, 9],
            [2, 4, 8, 6],
            [6, 8, 12, 10],
            [1, 2, 6, 5],
            [5, 6, 10, 9],
            [3, 4, 8, 7],
            [7, 8, 12, 11],
            [1, 2, 4, 3],
        ]
        self.assertEqual(fc.tolist(), expected_quad)

    def test_binsurface_0(self):
        no = binsurface(self.im, 0)[0]
        expected_mask = [[[0, 0], [0, -1]], [[0, 0], [0, -1]]]
        self.assertEqual(no.tolist(), expected_mask)

    def test_v2s(self):
        no, fc, _, _ = v2s(self.im, 0.5, 0.03)
        self.assertAlmostEqual(sum(elemvolume(no, fc[:, :3])), 5.01, 2)

    def test_v2m(self):
        no, el, fc = v2m(self.im, 0.5, 0.03, 10)
        self.assertAlmostEqual(round(sum(elemvolume(no, fc[:, :3])), 2), 5.01)
        self.assertAlmostEqual(round(sum(elemvolume(no, el[:, :4])), 4), 0.8786)

    def test_cgalv2m(self):
        no, el, fc = v2m(
            self.im.astype(np.uint8),
            [],
            {"radbound": 0.03, "distbound": 0.2},
            10,
            "cgalmesh",
        )
        self.assertEqual(removedupelem(fc[:, :3]).shape[0], 0)
        self.assertAlmostEqual(sum(elemvolume(no[:, :3], el[:, :4])), 0.7455, 3)

    def test_v2s_label(self):
        no, fc, _, _ = v2s(self.mask, 0.5, 0.5)
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], fc[:, :3])) * 0.001, 5.946015081617472, 2
        )

    def test_v2s_grayscale(self):
        no, fc, _, _ = v2s(self.dist, [200, 400], 1)
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], fc[:, :3])) * 0.0001, 4.816236473144449, 3
        )

    def test_finddisconnsurf(self):
        no, fc, _, _ = v2s(self.dist, [90, 200, 400], 5)
        fcs = finddisconnsurf(fc[:, :3])
        self.assertEqual(len(fcs), 5)

    def test_v2m_cgalmesh(self):
        no, el, _ = v2m(
            self.mask.astype(np.uint8),
            [],
            {"radbound": 1, "distbound": 0.5},
            5,
            "cgalmesh",
        )
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], el[:, :4])) * 0.0001, 3.4780013268627226, 2
        )

    def test_v2m_simplify(self):
        no, el, _ = v2m(
            self.mask.astype(np.uint8), 1.5, {"keepratio": 0.5}, 10, "simplify"
        )
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], el[:, :4])) * 0.001, 5.456390624999979, 2
        )

    def test_s2m_plc(self):
        no, el, fc = s2m(self.plcno, self.plcfc, 1, 3)
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], el[:, :4])), 213.33333333333337, 9
        )
        self.assertAlmostEqual(
            sum(elemvolume(no[:, :3], fc[:, :3])), 412.8904758569056, 4
        )


@unittest.skipIf(
    (int(mpl_ver[0]), int(mpl_ver[1])) < (3, 6), "Requires Matplotlib 3.6 or higher"
)
class Test_plot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_plot, self).__init__(*args, **kwargs)
        self.no, self.el = meshgrid6(
            np.arange(1, 3), np.arange(-1, 1), np.arange(2, 3.5, 0.5)
        )
        self.fc = volface(self.el)[0]

    def test_plotmesh_node(self):
        ax = plotmesh(self.no, "ro", hold="on")
        xy = ax["obj"][-1]._xy
        expected_fc = [
            [1.0, -1.0],
            [2.0, -1.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, -1.0],
            [2.0, -1.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [1.0, -1.0],
            [2.0, -1.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]

        self.assertEqual(xy.tolist(), expected_fc)

    def test_plotmesh_face(self):
        ax = plotmesh(self.no, self.fc, hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [9.0, 8.6803, 9.0, 20.0]

        self.assertEqual(len(facecolors), 20)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)

    def test_plotmesh_elem(self):
        ax = plotmesh(self.no, self.el, hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [9.0, 8.6803, 9.0, 20.0]

        self.assertEqual(len(facecolors), 20)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)

    def test_plotmesh_elemlabel(self):
        ax = plotmesh(
            self.no,
            np.hstack((self.el, np.ones(self.el.shape[0], dtype=int).reshape(-1, 1))),
            hold="on",
        )
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = (1, 4)

        self.assertEqual(len(facecolors), 20)
        self.assertEqual(np.unique(facecolors, axis=0).shape, expected_fc)

    def test_plotmesh_facelabel(self):
        ax = plotmesh(
            self.no,
            np.hstack((self.fc, np.array([1, 2] * 10).reshape(-1, 1))),
            None,
            hold="on",
        )
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = (2, 4)

        self.assertEqual(len(facecolors), 20)
        self.assertEqual(np.unique(facecolors, axis=0).shape, expected_fc)

    def test_plotmesh_elemnodeval(self):
        ax = plotmesh(self.no[:, [0, 1, 2, 0]], self.el, hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [8.0, 10.4074, 8.0, 20.0]

        self.assertEqual(len(facecolors), 20)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)

    def test_plotmesh_facenodeval(self):
        ax = plotmesh(self.no[:, [0, 1, 2, 0]], self.fc, "z < 3", hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [7.0, 8.6728, 7.0, 18.0]

        self.assertEqual(len(facecolors), 18)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)

    def test_plotmesh_selector(self):
        ax = plotmesh(self.no[:, [0, 1, 2, 0]], self.fc, "(z < 3) & (x < 2)", hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [4.8877, 5.0, 4.451, 14.0]

        self.assertEqual(len(facecolors), 14)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)

    def test_plotmesh_elemselector(self):
        ax = plotmesh(self.no, self.fc, "z < 2.5", hold="on")
        facecolors = np.array(ax["obj"][-1].get_facecolors())
        expected_fc = [3.9102, 4.0, 2.9608, 10.0]

        self.assertEqual(len(facecolors), 10)
        self.assertEqual(np.round(np.sum(facecolors, axis=0), 4).tolist(), expected_fc)


if __name__ == "__main__":
    unittest.main()
