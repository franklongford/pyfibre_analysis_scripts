import os
from unittest import TestCase

from pyfibre_analysis_tools.analysis_tools import load_databases
from pyfibre_analysis_tools.tests import (
    example_hdf5_path, example_excel_path)


class TestAnalysisTools(TestCase):

    def setUp(self):
        self.expected_data = {
            "File": 'test-pyfibre',
            "Label": 1,
            "Group": 'fixtures',
            'Cell Segment Area': 3356.08112479781,
            'Cell Segment Circularity': 0.6064908120463052,
            'Cell Segment Coverage': 0.40185,
            'Cell Segment Eccentricity': 0.8047600284349302,
            'Cell Segment PL Angle SDI': 0.04126322992698592,
            'Cell Segment PL Coherence': 0.050135506739157745,
            'Cell Segment PL Entropy': 5.990253269898627,
            'Cell Segment PL Local Coherence': 0.5695887244727482,
            'Cell Segment PL Mean': 14.685185185185187,
            'Cell Segment PL STD': 8.130639297218199,
            'Fibre Angle SDI': 0.17333333333333453,
            'Fibre Network Connectivity': 6.593060330942346,
            "Fibre Network Cross-Link Density": 0.8292806484295846,
            "Fibre Network Degree": 0.06835129756965094,
            "Fibre Network Eigenvalue": 1.8847226182984607,
            "Fibre Segment Area": 22302.0,
            "Fibre Segment Circularity": 0.13743224855487896,
            "Fibre Segment Coverage": 0.55755,
            "Fibre Segment Eccentricity": 0.5285029950922379,
            "Fibre Segment SHG Angle SDI": 0.03213121398510797,
            "Fibre Segment SHG Coherence": 0.03249977696095154,
            "Fibre Segment SHG Entropy": 6.656214652841974,
            "Fibre Segment SHG Local Coherence": 0.5648194626697063,
            "Fibre Segment SHG Mean": 18.32043463964368,
            "Fibre Segment SHG STD": 10.700801248848675,
            "Mean Fibre Length": 25.87370244918227,
            "Mean Fibre Waviness": 0.9262702557385855,
            "No. Cells": 9,
            "No. Fibres": 104
        }

    def test_load_databases_h5(self):
        data_dir, filename = os.path.split(example_hdf5_path)

        database = load_databases(filename, [data_dir])
        self.assertEqual(1, len(database))

        for key, value in self.expected_data.items():
            with self.subTest(key):
                if isinstance(value, (str, bool)):
                    self.assertEqual(value, database.iloc[0][key])
                else:
                    self.assertAlmostEqual(value, database.iloc[0][key])

        filename, ext = os.path.splitext(filename)
        database = load_databases(filename, [data_dir], ext=ext)
        self.assertEqual(1, len(database))

    def test_load_databases_xls(self):
        data_dir, filename = os.path.split(example_excel_path)

        database = load_databases(filename, [data_dir])
        self.assertEqual(1, len(database))

        for key, value in self.expected_data.items():
            with self.subTest(key):
                if isinstance(value, (str, bool)):
                    self.assertEqual(value, database.iloc[0][key])
                else:
                    self.assertAlmostEqual(value, database.iloc[0][key])

        filename, ext = os.path.splitext(filename)
        database = load_databases(filename, [data_dir], ext=ext)
        self.assertEqual(1, len(database))
