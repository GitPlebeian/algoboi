//
//  BKTestScoreModel.swift
//  stockInterface
//
//  Created by CHONK on 2/7/24.
//

import Foundation

struct BKTestScoreModel: Codable {
    let predictedPercentageGain: Float
    let currentClosingPrice:     Float
    let ticker:                  String
    let companyName:             String
}
