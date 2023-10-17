//
//  MLTrainingData.swift
//  stockInterface
//
//  Created by CHONK on 10/17/23.
//

import Foundation

struct MLTrainingData: Codable {
    
    var slopeOf9DayEMA: [Float]
    var slopeOf25DayEMA: [Float]
    var closes: [Float]
    var minimumStartIndex: Int = 25
}
