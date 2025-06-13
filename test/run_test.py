import unittest
import numpy as np

import sys
import os

# Add project_root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iso2mesh import *


class Test_primitives(unittest.TestCase):
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
        self.assertAlmostEqual(round(sum(elemvolume(no, el[:, :4])), 4), 4.1802)

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

    def test_meshacylinderplc(self):
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


class Test_utils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_utils, self).__init__(*args, **kwargs)
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

    def test_uniqedges(self):
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
        self.assertEqual(uniqedges(self.el)[0].tolist(), expected)

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

    def test_surfvolume(self):
        self.assertEqual(surfvolume(self.no, self.fc), 1)

    def test_insurface(self):
        pts = np.array([[1.5, -0.9, 2.1], [1, 0, 2], [-1, 0, 2], [1.2, 0, 2.5]])
        expected = [1, 1, 0, 1]
        result = insurface(self.no, self.fc, pts).tolist()
        self.assertEqual(result, expected)


class Test_surfaces(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_surfaces, self).__init__(*args, **kwargs)
        self.no, self.fc, self.el = meshacylinder(
            [1, 1, 1], [2, 3, 4], [0.5, 0.7], 1, 100, 8
        )
        self.el = meshreorient(self.no, self.el[:, :4])[0]
        self.no1, self.fc1 = meshcheckrepair(self.no, self.fc, "deep")

    def test_removeisolatednode(self):
        tmp = np.mean(removeisolatednode(self.no, self.fc)[0], axis=0)
        print(np.linalg.norm(tmp - np.array([1.44, 1.8799, 2.3198])))
        self.assertTrue(np.linalg.norm(tmp - np.array([1.44, 1.8799, 2.3198])) < 0.01)

    def test_meshreorient(self):
        self.assertEqual(
            meshreorient(self.no, self.el[:, [0, 1, 3, 2]])[0].tolist(),
            meshreorient(self.no, self.el[:, :4])[0].tolist(),
        )

    def test_removedupnodes(self):
        el1 = self.el.copy()
        mask = self.el[:, :4] < 2
        el1[:, :4][mask] = self.no.shape[0] + el1[:, :4][mask]
        print(
            np.linalg.norm(
                np.mean(
                    removedupnodes(np.vstack([self.no, self.no[:2]]), el1[:, :4])[0],
                    axis=0,
                )
                - np.array([1.43802, 1.875757, 2.31368])
            )
        )
        self.assertTrue(
            np.linalg.norm(
                np.mean(
                    removedupnodes(np.vstack([self.no, self.no[:2]]), el1[:, :4])[0],
                    axis=0,
                )
                - np.array([1.43802, 1.875757, 2.31368])
            )
            < 0.1
        )

    def test_removedupelem(self):
        print(removedupelem(np.vstack([self.el[:, :4], self.el[:-5, :4]])).tolist())
        self.assertEqual(
            removedupelem(np.vstack([self.el[:, :4], self.el[:-5, :4]])).tolist(),
            self.el[-5:, :4].tolist(),
        )

    def test_surfreorient(self):
        no1, fc1 = surfreorient(self.no, self.fc)
        self.assertEqual(fc.shape, fc1.shape)
        self.assertFalse(np.any(elemvolume(no1, fc1) <= 0))

    def test_meshcheckrepairdeep(self):
        self.assertEqual(list(self.no1.shape), [50, 3])
        self.assertFalse(np.any(elemvolume(self.no1, self.fc1) <= 0))

    def test_meshcheckrepairmeshfix(self):
        no2, fc2 = meshcheckrepair(self.no, self.fc[4:, :], "meshfix")
        print([sum(elemvolume(self.no1, self.fc1)), sum(elemvolume(no2, fc2))])
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


if __name__ == "__main__":
    unittest.main()
