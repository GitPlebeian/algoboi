//
//  MLPlayback.swift
//  stockInterface
//
//  Created by CHONK on 10/24/23.
//

import Foundation

struct MLPlayback: Codable {
    
    var episodeCount: Int
    var length: Int
    var purchaseIndexs: [[Int]] // Episode -> Index
    var sellIndexs:     [[Int]] // Episode -> Index
}
