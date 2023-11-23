//
//  IndicatorData.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

struct IndicatorData: Codable {
    let ticker: String
    var length: Int {
        return sma200.count
    }
    let sma200: [Float]
    let sma50:  [Float]
    let ema14:  [Float]
    let ema28:  [Float]
}
