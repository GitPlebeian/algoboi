//
//  TickerNameModel.swift
//  stockInterface
//
//  Created by CHONK on 11/20/23.
//

import Foundation

struct TickerNameModel: Codable {
    let symbol: String
    let name: String
    
    init(symbol: String, name: String) {
        self.symbol = symbol
        self.name = name
    }
    
    init?(dict: [String: Any]) {
        guard let symbol = dict["ticker"] as? String else {return nil}
        guard let name = dict["name"] as? String else {return nil}
        guard let type = dict["type"] as? String else {return nil}
        guard let exchange = dict["primary_exchange"] as? String else {return nil}
        guard let isActive = dict["active"] as? Bool else {return nil}
        guard let currencyName = dict["currency_name"] as? String else {return nil}
        guard let locale = dict["locale"] as? String else {return nil}
        guard let market = dict["market"] as? String else {return nil}
        
        if (exchange != "XNYS" && exchange != "XNAS") || type != "CS" || currencyName != "usd" || market != "stocks" || isActive != true || locale != "us" {
            return nil
        }
        
        let bannedWords: [String] = ["Pharmaceuticals", "Acquisition", "Therapeutics", "International", "%", "bio", "Bio", "Healthcare", "TRUST", "Trust", "Capital", "REIT", "Reit"]
        for e in bannedWords {
            if name.contains(e) {
                print("Banning: \(name)")
                return nil
            }
        }
        self.init(symbol: symbol, name: name)
    }
}
