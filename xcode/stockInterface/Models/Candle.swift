//
//  Candle.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Foundation

struct Candle: Codable {
    
    let volume: Int64
    let volumeWeighted: Float
    let timestamp: Date
    let transactionCount: Int64
    let open: Float
    let close: Float
    let high: Float
    let low: Float
}
