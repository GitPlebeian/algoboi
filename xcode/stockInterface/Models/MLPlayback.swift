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
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.purchaseIndexs = try container.decode([[Int]].self, forKey: .purchaseIndexs)
        self.sellIndexs = try container.decode([[Int]].self, forKey: .sellIndexs)
        self.length = try container.decode(Int.self, forKey: .length)
//        self.episodeCount = try container.decode(Int.self, forKey: .episodeCount)
        self.episodeCount = self.purchaseIndexs.count
    }
}
